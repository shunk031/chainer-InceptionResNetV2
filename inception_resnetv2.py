import chainer
import chainer.links as L
import chainer.functions as F


class BasicConv2D(chainer.Chain):

    def __init__(self, in_places, out_places, ksize, stride, padding=0):
        super(BasicConv2D, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_places,
                                        out_places,
                                        ksize=ksize,
                                        stride=stride,
                                        pad=padding,
                                        nobias=True)
            self.bn = L.BatchNormalization(out_places)

    def __call__(self, x):
        h = F.relu(self.bn(self.conv(x)))

        return h


class Mixed5b(chainer.Chain):

    def __init__(self):
        super(Mixed5b, self).__init__()
        with self.init_scope():
            self.branch0 = BasicConv2D(192, 96, ksize=1, stride=1)

            self.branch1_0 = BasicConv2D(192, 48, ksize=1, stride=1)
            self.branch1_1 = BasicConv2D(48, 64, ksize=5, stride=1, padding=2)

            self.branch2_0 = BasicConv2D(192, 64, ksize=1, stride=1)
            self.branch2_1 = BasicConv2D(64, 96, ksize=3, stride=1, padding=1)
            self.branch2_2 = BasicConv2D(96, 96, ksize=3, stride=1, padding=1)

            self.branch3 = BasicConv2D(192, 64, ksize=1, stride=1)

    def __call__(self, x):

        h0 = self.branch0(x)

        h1 = self.branch1_0(x)
        h1 = self.branch1_1(h1)

        h2 = self.branch2_0(x)
        h2 = self.branch2_1(h2)
        h2 = self.branch2_2(h2)

        h3 = self.branch3(x)
        h = F.concat((h0, h1, h2, h3), axis=1)

        return h


class Block35(chainer.Chain):

    def __init__(self, scale=1.0):
        self.scale = scale

        super(Block35, self).__init__()
        with self.init_scope():
            self.branch0 = BasicConv2D(320, 32, ksize=1, stride=1)

            self.branch1_0 = BasicConv2D(320, 32, ksize=1, stride=1)
            self.branch1_1 = BasicConv2D(32, 32, ksize=3, stride=1, padding=1)

            self.branch2_0 = BasicConv2D(320, 32, ksize=1, stride=1)
            self.branch2_1 = BasicConv2D(32, 48, ksize=3, stride=1, padding=1)
            self.branch2_2 = BasicConv2D(48, 64, ksize=3, stride=1, padding=1)

            self.conv2 = L.Convolution2D(128, 320, ksize=1, stride=1)

    def __call__(self, x):

        h0 = self.branch0(x)

        h1 = self.branch1_0(x)
        h1 = self.branch1_1(h1)

        h2 = self.branch2_0(x)
        h2 = self.branch2_1(h2)
        h2 = self.branch2_2(h2)

        h = F.concat((h0, h1, h2), axis=1)
        h = self.conv2(h)
        h = F.relu(h * self.scale + x)

        return h


class Mixed6a(chainer.Chain):

    def __init__(self):
        super(Mixed6a, self).__init__()
        with self.init_scope():
            self.branch0 = BasicConv2D(320, 384, ksize=3, stride=2)

            self.branch1_0 = BasicConv2D(320, 256, ksize=1, stride=1)
            self.branch1_1 = BasicConv2D(256, 256, ksize=3, stride=1, padding=1)
            self.branch1_2 = BasicConv2D(256, 384, ksize=3, stride=2)

    def __call__(self, x):
        h0 = self.branch0(x)

        h1 = self.branch1_0(x)
        h1 = self.branch1_1(h1)
        h1 = self.branch1_2(h1)

        h2 = F.max_pooling_2d(x, ksize=3, stride=2, cover_all=False)

        h = F.concat((h0, h1, h2), axis=1)

        return h


class Block17(chainer.Chain):

    def __init__(self, scale=1.0):
        self.scale = scale

        super(Block17, self).__init__()
        with self.init_scope():
            self.branch0 = BasicConv2D(1088, 192, ksize=1, stride=1)

            self.branch1_0 = BasicConv2D(1088, 128, ksize=1, stride=1)
            self.branch1_1 = BasicConv2D(128, 160, ksize=(1, 7), stride=1, padding=(0, 3))
            self.branch1_2 = BasicConv2D(160, 192, ksize=(7, 1), stride=1, padding=(3, 0))

            self.conv = L.Convolution2D(384, 1088, ksize=1, stride=1)

    def __call__(self, x):
        h0 = self.branch0(x)

        h1 = self.branch1_0(x)
        h1 = self.branch1_1(h1)
        h1 = self.branch1_2(h1)

        h = F.concat((h0, h1), axis=1)
        h = self.conv(h)
        h = F.relu(h * self.scale + x)

        return h


class Mixed7a(chainer.Chain):

    def __init__(self):
        super(Mixed7a, self).__init__()
        with self.init_scope():
            self.branch0_0 = BasicConv2D(1088, 256, ksize=1, stride=1)
            self.branch0_1 = BasicConv2D(256, 384, ksize=3, stride=2)

            self.branch1_0 = BasicConv2D(1088, 256, ksize=1, stride=1)
            self.branch1_1 = BasicConv2D(256, 288, ksize=3, stride=2)

            self.branch2_0 = BasicConv2D(1088, 256, ksize=1, stride=1)
            self.branch2_1 = BasicConv2D(256, 288, ksize=3, stride=1, padding=1)
            self.branch2_2 = BasicConv2D(288, 320, ksize=3, stride=2)

    def __call__(self, x):

        h0 = self.branch0_0(x)
        h0 = self.branch0_1(h0)

        h1 = self.branch1_0(x)
        h1 = self.branch1_1(h1)

        h2 = self.branch2_0(x)
        h2 = self.branch2_1(h2)
        h2 = self.branch2_2(h2)

        h3 = F.max_pooling_2d(x, ksize=3, stride=2, cover_all=False)

        h = F.concat((h0, h1, h2, h3), axis=1)

        return h


class Block8(chainer.Chain):

    def __init__(self, scale=1.0, no_relu=False):
        self.scale = scale
        self.no_relu = no_relu

        super(Block8, self).__init__()
        with self.init_scope():
            self.branch0 = BasicConv2D(2080, 192, ksize=1, stride=1)

            self.branch1_0 = BasicConv2D(2080, 192, ksize=1, stride=1)
            self.branch1_1 = BasicConv2D(192, 224, ksize=(1, 3), stride=1, padding=(0, 1))
            self.branch1_2 = BasicConv2D(224, 256, ksize=(3, 1), stride=1, padding=(1, 0))

            self.conv = L.Convolution2D(448, 2080, ksize=1, stride=1)

    def __call__(self, x):

        h0 = self.branch0(x)

        h1 = self.branch1_0(x)
        h1 = self.branch1_1(h1)
        h1 = self.branch1_2(h1)

        h = F.concat((h0, h1), axis=1)
        h = self.conv(h)
        h = h * self.scale + x

        if not self.no_relu:
            h = F.relu(h)

        return h


class InceptionResNetV2(chainer.Chain):

    def __init__(self, num_classes=55):
        self.input_space = None

        super(InceptionResNetV2, self).__init__()
        with self.init_scope():
            self.conv_1a = BasicConv2D(3, 32, ksize=3, stride=2)
            self.conv_2a = BasicConv2D(32, 32, ksize=3, stride=1)
            self.conv_2b = BasicConv2D(32, 64, ksize=3, stride=1, padding=1)
            self.conv_3b = BasicConv2D(64, 80, ksize=1, stride=1)
            self.conv_4a = BasicConv2D(80, 192, ksize=3, stride=1)
            self.mixed_5b = Mixed5b()

            for i in range(10):
                name = 'repeat{}'.format(i + 1)
                setattr(self, name, Block35(scale=0.17))

            self.mixed_6a = Mixed6a()
            for i in range(20):
                name = 'repeat1_{}'.format(i + 1)
                setattr(self, name, Block17(scale=0.10))

            self.mixed_7a = Mixed7a()
            for i in range(9):
                name = 'repeat2_{}'.format(i + 1)
                setattr(self, name, Block8(scale=0.20))

            self.block8 = Block8(no_relu=True)
            self.conv_7b = BasicConv2D(2080, 1536, ksize=1, stride=1)
            self.fc = L.Linear(1536, num_classes)

    def __call__(self, x):
        h = self.conv_1a(x)
        h = self.conv_2a(h)
        h = self.conv_2b(h)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = self.conv_3b(h)
        h = self.conv_4a(h)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = self.mixed_5b(h)
        for i in range(10):
            h = getattr(self, 'repeat{}'.format(i + 1))(h)
        h = self.mixed_6a(h)
        for i in range(20):
            h = getattr(self, 'repeat1_{}'.format(i + 1))(h)
        h = self.mixed_7a(h)
        for i in range(9):
            h = getattr(self, 'repeat2_{}'.format(i + 1))(h)
        h = self.block8(h)
        h = self.conv_7b(h)
        h = _global_average_pooling_2d(h)
        h = self.fc(h)

        return h


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = F.reshape(h, (n, channel))
    return h
