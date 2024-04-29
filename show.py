import sys
import os
import subprocess
sys.path.append('../')
from pycore.tikzeng import *


# 定义神经网络架构
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    to_input("1.jpeg", to='(-5,0,0)', width=6, height=6, name="temp"),
    to_input("2.jpeg", to='(-4,0,0)', width=6, height=6, name="temp"),
    to_input("3.jpeg", to='(-3,0,0)', width=6, height=6, name="temp"),
    to_input("4.jpeg", to='(-2,0,0)', width=6, height=6, name="temp"),
    to_input("5.jpeg", to='(-1,0,0)', width=6, height=6, name="temp"),

    # s_filer表示该层的图像大小 (需要自己计算),n_filer表示输入通道和输出通道大小 (自己设定)
    to_Conv("conv1", s_filer=256, n_filer=3, offset="(0,0,0)", to="(0,0,0)", height=50, depth=50, width=3, caption='CONV1'),
    to_Conv("conv2", s_filer=256, n_filer=64, offset="(0,0,0)", to="(conv1-east)", height=50, depth=50, width=3, caption=''),
    to_Pool("pool1", offset="(0,0,0)", to="(conv2-east)", height=30, depth=30, width=3, caption=""),

    to_Conv("conv3", s_filer=128, n_filer=64, offset="(3,0,0)", to="(pool1-east)", height=30, depth=30, width=3,caption='CONV2'),
    to_connection("pool1", "conv3"),
    to_Conv("conv4", s_filer=128, n_filer=128, offset="(0,0,0)", to="(conv3-east)", height=30, depth=30, width=3,caption=''),
    to_Pool("pool2", offset="(0,0,0)", to="(conv4-east)", height=18, depth=18, width=3, caption=""),

    to_Conv("conv5", s_filer=64, n_filer=128, offset="(3,0,0)", to="(pool2-east)", height=30, depth=30, width=3,caption=''),
    to_connection("pool2", "conv5"),
    to_Conv("conv6", s_filer=64, n_filer=256, offset="(0,0,0)", to="(conv5-east)", height=30, depth=30, width=3,caption='CONV3'),
    to_Conv("conv7", s_filer=64, n_filer=256, offset="(0,0,0)", to="(conv6-east)", height=30, depth=30, width=3,caption=''),
    to_Pool("pool3", offset="(0,0,0)", to="(conv7-east)", height=16, depth=16, width=3, caption=""),

    to_Conv("conv8", s_filer=32, n_filer=256, offset="(3,0,0)", to="(pool3-east)", height=16, depth=16, width=3,caption=''),
    to_connection("pool3", "conv8"),
    to_Conv("conv9", s_filer=32, n_filer=512, offset="(0,0,0)", to="(conv8-east)", height=16, depth=16, width=3,caption='CONV4'),
    to_Conv("conv10", s_filer=32, n_filer=512, offset="(0,0,0)", to="(conv9-east)", height=16, depth=16, width=3,caption=''),
    to_Pool("pool4", offset="(0,0,0)", to="(conv10-east)", height=8, depth=8, width=3, caption=""),

    to_Conv("conv11", s_filer=16, n_filer=512, offset="(3,0,0)", to="(pool4-east)", height=8, depth=8, width=3,caption=''),
    to_connection("pool4", "conv11"),
    to_Conv("conv12", s_filer=16, n_filer=512, offset="(0,0,0)", to="(conv11-east)", height=8, depth=8, width=3, caption='CONV5'),
    to_Conv("conv13", s_filer=16, n_filer=512, offset="(0,0,0)", to="(conv12-east)", height=8, depth=8, width=3, caption=''),
    to_Pool("pool5", offset="(0,0,0)", to="(conv13-east)", height=6, depth=6, width=3, caption=""),

    to_Pool("pool6", offset="(2,0,0)", to="(pool5-east)", height=6, depth=6, width=3, caption="AdaptiveAvgPool"),
    to_connection("pool5", "pool6"),

    to_SoftMax(name='fc1', s_filer=4096, offset="(4,0,0)", to="(pool6-east)", width=1.5, height=1.5, depth=100,
               opacity=0.8, caption='FC1'),
    to_connection("pool6", "fc1"),

    to_SoftMax(name='fc2', s_filer=4096, offset="(2,0,0)", to="(fc1-east)", width=1.5, height=1.5, depth=50,
               opacity=0.8, caption='FC2'),
    to_connection("fc1", "fc2"),

    to_SoftMax(name='fc3', s_filer=5, offset="(2,0,0)", to="(fc2-east)", width=1.5, height=1.5, depth=5,
               opacity=0.8, caption='FC3'),
    to_connection("fc2", "fc3"),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

    # 使用 LaTeX 编译器将 .tex 文件转换为 .pdf 文件
    subprocess.call([r'D:\MiKTeX\install\miktex\bin\x64\pdflatex.exe', namefile + '.tex'])

    pdf_file = namefile + '.pdf'
    image_file = namefile + '.png'

    subprocess.call([r'D:\ghostscript\gs10.01.1\bin\gswin64c.exe', '-sDEVICE=pngalpha', '-o', image_file, '-r300', pdf_file])

    # 删除中间生成的文件
    cleanup(namefile)


def cleanup(namefile):
    # 删除中间生成的文件
    extensions = ['.aux', '.log', '.tex']
    for ext in extensions:
        filename = namefile + ext
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == '__main__':
    main()