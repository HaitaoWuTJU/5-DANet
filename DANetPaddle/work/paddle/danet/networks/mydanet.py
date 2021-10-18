import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable

from paddle.fluid.dygraph import Conv2D, Pool2D, Dropout, BatchNorm, Sequential
from paddle.fluid.layers import bmm,image_resize,create_parameter, reduce_max, reshape, transpose,softmax,expand_as

# ResNet中使用了BatchNorm层，在卷积层的后面加上BatchNorm以提升数值稳定性
# 定义卷积批归一化块
class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        """
        
        num_channels, 卷积层的输入通道数
        num_filters, 卷积层的输出通道数
        stride, 卷积层的步幅
        groups, 分组卷积的组数，默认groups=1不使用分组卷积
        act, 激活函数类型，默认act=None不使用激活函数
        """
        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        # 创建BatchNorm层
        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y

# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()
        # 创建第一个卷积层 1x1
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        # 创建第二个卷积层 3x3
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        # 创建第三个卷积 1x1，但输出通道数乘以4
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        # 如果conv2的输出跟此残差块的输入数据形状一致，则shortcut=True
        # 否则shortcut = False，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        # 如果shortcut=True，直接将inputs跟conv2的输出相加
        # 否则需要对inputs进行一次卷积，将形状调整成跟conv2输出一致
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv2)
        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(y)

# 定义ResNet模型
class ResNet(fluid.dygraph.Layer):
    def __init__(self, layers=101, class_dim=1):
        """
        
        layers, 网络层数，可以是50, 101或者152
        class_dim，分类标签的类别数
        """
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            #ResNet50包含多个模块，其中第2到第5个模块分别包含3、4、6、3个残差块
            depth = [3, 4, 6, 3]
        elif layers == 101:
            #ResNet101包含多个模块，其中第2到第5个模块分别包含3、4、23、3个残差块
            depth = [3, 4, 23, 3]
        elif layers == 152:
            #ResNet50包含多个模块，其中第2到第5个模块分别包含3、8、36、3个残差块
            depth = [3, 8, 36, 3]
        
        # 残差块中使用到的卷积的输出通道数
        num_filters = [64, 128, 256, 512]
        # num_filters = [32, 64, 128, 256]

        # ResNet的第一个模块，包含1个7x7卷积，后面跟着1个最大池化层
        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = Pool2D(
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        # ResNet的第二到第五个模块c2、c3、c4、c5
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1, # c3、c4、c5将会在第一个残差块使用stride=2；其余所有残差块stride=1
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 在c5的输出特征图上使用全局池化
        self.pool2d_avg = Pool2D(pool_size=7, pool_type='avg', global_pooling=True)

        # stdv用来作为全连接层随机初始化参数的方差
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        
        # 创建全连接层，输出大小为类别数目
        self.out = Linear(input_dim=2048, output_dim=class_dim,
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.Uniform(-stdv, stdv)))

        
    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        # y = self.pool2d_avg(y)
        # y = fluid.layers.reshape(y, [y.shape[0], -1])
        # y = self.out(y)
        return y


class DANet(fluid.dygraph.Layer):
    def __init__(self,name_scope,out_chs=20,in_chs=1024,inter_chs=512):
        super(DANet,self).__init__(name_scope)
        name_scope = self.full_name()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.inter_chs = inter_chs if inter_chs else in_chs


        self.backbone = ResNet(101)
        self.conv5p = Sequential(
            Conv2D(self.in_chs, self.inter_chs, 3, padding=1),
            BatchNorm(self.inter_chs,act='relu'),
        )
        self.conv5c = Sequential(
            Conv2D(self.in_chs, self.inter_chs, 3, padding=1),
            BatchNorm(self.inter_chs,act='relu'),
        )

        self.sp = PAM_module(self.inter_chs)
        self.sc = CAM_module(self.inter_chs)

        self.conv6p = Sequential(
            Conv2D(self.inter_chs, self.inter_chs, 3, padding=1),
            BatchNorm(self.inter_chs,act='relu'),
        )
        self.conv6c = Sequential(
            Conv2D(self.inter_chs, self.inter_chs, 3, padding=1),
            BatchNorm(self.inter_chs,act='relu'),
        )

        self.conv7p = Sequential(
            Dropout(0.1),
            Conv2D(self.inter_chs, self.out_chs, 1),
        )
        self.conv7c = Sequential(
            Dropout(0.1),
            Conv2D(self.inter_chs, self.out_chs, 1),
        )
        self.conv7pc = Sequential(
            Dropout(0.1),
            Conv2D(self.inter_chs, self.out_chs, 1),
        )

    def forward(self,x):

        feature = self.backbone(x)

        p_f = self.conv5p(feature)
        p_f = self.sp(p_f)
        p_f = self.conv6p(p_f)
        p_out = self.conv7p(p_f)

        c_f = self.conv5c(feature)
        c_f = self.sc(c_f)
        c_f = self.conv6c(c_f)
        c_out = self.conv7c(c_f)

        sum_f = p_f+c_f
        sum_out = self.conv7pc(sum_f)

        p_out = image_resize(p_out,out_shape=x.shape[2:])
        c_out = image_resize(c_out,out_shape=x.shape[2:])
        sum_out = image_resize(sum_out,out_shape=x.shape[2:])
        return [p_out, c_out, sum_out]
        # return sum_out

class PAM_module(fluid.dygraph.Layer):
    def __init__(self,in_chs,inter_chs=None):
        super(PAM_module,self).__init__()
        self.in_chs = in_chs
        self.inter_chs = inter_chs if inter_chs else in_chs
        self.conv_query = Conv2D(self.in_chs,self.inter_chs,1)
        self.conv_key = Conv2D(self.in_chs,self.inter_chs,1)
        self.conv_value = Conv2D(self.in_chs,self.inter_chs,1)
        self.gamma = create_parameter([1], dtype='float32')
    
    def forward(self,x):
        b,c,h,w = x.shape

        f_query = self.conv_query(x)
        f_query = reshape(f_query,(b, -1, h*w))
        f_query = transpose(f_query,(0, 2, 1)) 

        f_key = self.conv_key(x)
        f_key = reshape(f_key,(b, -1, h*w))

        f_value = self.conv_value(x)
        f_value = reshape(f_value,(b, -1, h*w))
        f_value = transpose(f_value,(0, 2, 1)) 


        f_similarity = bmm(f_query, f_key)                        # [h*w, h*w]
        f_similarity = softmax(f_similarity)
        f_similarity = transpose(f_similarity,(0, 2, 1))

        f_attention = bmm(f_similarity, f_value)                        # [h*w, c]
        f_attention = reshape(f_attention,(b,c,h,w))

        out = self.gamma*f_attention + x
        return out

class CAM_module(fluid.dygraph.Layer):
    def __init__(self,in_chs,inter_chs=None):
        super(CAM_module,self).__init__()
        self.in_chs = in_chs
        self.inter_chs = inter_chs if inter_chs else in_chs
        self.gamma = create_parameter([1], dtype='float32')

    def forward(self,x):
        b,c,h,w = x.shape

        f_query = reshape(x,(b, -1, h*w))
        f_key = reshape(x,(b, -1, h*w))
        f_key = transpose(f_key,(0, 2, 1)) 
        f_value = reshape(x,(b, -1, h*w))

        f_similarity = bmm(f_query, f_key)                        # [h*w, h*w]
        f_similarity_max = reduce_max(f_similarity, -1, keep_dim=True)
        f_similarity_max_reshape = expand_as(f_similarity_max,f_similarity)
        f_similarity = f_similarity_max_reshape-f_similarity

        f_similarity = softmax(f_similarity)
        f_similarity = transpose(f_similarity,(0, 2, 1)) 

        f_attention = bmm(f_similarity,f_value)                        # [h*w, c]
        f_attention = reshape(f_attention,(b,c,h,w))

        out = self.gamma*f_attention + x
        return out



