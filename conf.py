
args = None

CIFAR10Opt = {
    'name': 'cifar10',
    'batch_size': 64, # 128

    'learning_rate': 0.1,  # initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'img_size': 32,

    'file_path': './dataset/CIFAR-10-C',
    'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'num_class': 10,
    'severity': 5,
    'domains': ["original",

                "test",

                "gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],
    'src_domains': ["original"],
    'tgt_domains': [
                    "gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",

    ],
}

CIFAR100Opt = {
    'name': 'cifar100',
    'batch_size': 128,

    'learning_rate': 0.1,  # initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'img_size': 32,

    'file_path': './dataset/CIFAR-100-C',
    'classes': ['beaver', 'dolphin', 'otter', 'seal', 'whale',
                'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
                'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                'bottles', 'bowls', 'cans', 'cups', 'plates',
                'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
                'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                'bed', 'chair', 'couch', 'table', 'wardrobe',
                'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                'bear', 'leopard', 'lion', 'tiger', 'wolf',
                'bridge', 'castle', 'house', 'road', 'skyscraper',
                'cloud', 'forest', 'mountain', 'plain', 'sea',
                'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                'crab', 'lobster', 'snail', 'spider', 'worm',
                'baby', 'boy', 'girl', 'man', 'woman',
                'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                'maple', 'oak', 'palm', 'pine', 'willow',
                'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
                'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'],
    'num_class': 100,
    'severity': 5,
    # 'corruptions': ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur",
    #                 "brightness", "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
    #                 "jpeg_compression", "elastic_transform"],
    'domains': ["original",

                "test",

                "gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],
    'src_domains': ["original"],
    'tgt_domains': [
        "gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",

    ],
}


IMAGENET_C = {
    # referred to for hyperparams: https://github.com/Lornatang/ResNet-PyTorch/blob/9e529757ce0607aafeae2ddd97142201b3d4cadd/examples/imagenet/main.py
    'name': 'imagenet',
    'batch_size': 256,
    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 224,

    'file_path': './dataset/ImageNet-C',
    'num_class': 1000,
    'severity': 5,
    'domains': ["gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],

    'src_domains': ["original"],
    'tgt_domains': ["gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",

    ],
}

PACSOpt = {
    'name': 'pacs',
    'batch_size': 64,

    'learning_rate': 0.001,  # initial learning rate
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 224,

    'file_path': './domainbed_dataset/PACS',
    'classes': ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'],
    'num_class': 7,
    'severity': None,
    'domains': ["art_painting", "cartoon", "photo", "sketch"],
    'src_domains': ["photo"],
    'tgt_domains': ["art_painting", "cartoon", "sketch"],
}

VLCSOpt = {
    'name': 'vlc',
    'batch_size': 64,

    'learning_rate': 0.001,  # initial learning rate
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 224,

    'file_path': './domainbed_dataset/VLCS',
    'classes': ['bird', 'car', 'chair', 'dog', 'person'],
    'num_class': 5,
    'severity': None,
    'domains': ["Caltech101", "LabelMe", "SUN09", "VOC2007"],
    'src_domains': ["Caltech101"],
    'tgt_domains': ["LabelMe", "SUN09", "VOC2007"],
}

DOMAINNET126Opt = {
    'name': 'domainnet-126',
    'batch_size': 64,

    'learning_rate': 0.001,  # initial learning rate
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 224,

    'file_path': './dataset/domainnet',
    'classes': ['bird', 'car', 'chair', 'dog', 'person'],
    'num_class': 126,
    'severity': None,
    'domains': ["clipart", "painting", "real", "sketch"],
    'src_domains': ["real"],
    'tgt_domains': ["clipart", "painting", "sketch"],
}

TINYIMAGENET_C = {
    'name': 'tiny-imagenet',
    'batch_size': 100,
    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 224,

    'file_path': './dataset/Tiny-ImageNet-C',
    'num_class': 200,
    'severity': 5,
    'domains': ["gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],

    'src_domains': ["original"],
    'tgt_domains': ["gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",

    ],
}

IMAGENET_R = {
    # referred to for hyperparams: https://github.com/Lornatang/ResNet-PyTorch/blob/9e529757ce0607aafeae2ddd97142201b3d4cadd/examples/imagenet/main.py
    'name': 'imagenet-r',
    'batch_size': 256,
    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/imagenet-r',
    'num_class': 200, # select 200 from 1000
    'severity': 5,

    'domains': ['original', 'corrupt'], 

    'src_domains': ["original"],
    'tgt_domains': ["corrupt"],
    
    'indices_in_1k' : [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988]

}

COLORED_MNIST = {
    # referred to for hyperparams: https://github.com/Lornatang/ResNet-PyTorch/blob/9e529757ce0607aafeae2ddd97142201b3d4cadd/examples/imagenet/main.py
    'name': 'color-mnist',
    'batch_size': 256,
    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/color-mnist',
    'num_class': 2, # select 200 from 1000
    'severity': 5,

    'domains': ['train1', 'train2', 'all_train', 'test'], 

    'src_domains': ["all_train"],
    'tgt_domains': ["corrupt"],
    
    'indices_in_1k' : [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988]

}

CONT_SEQUENCE_PACS = {
    0 :  ["art_painting", "cartoon", "sketch"], # original
    2 :  ["art_painting", "sketch", "cartoon"],
    3 :  ["cartoon", "art_painting", "sketch"],
    4 :  ["cartoon", "sketch", "art_painting"],
    5 :  ["sketch", "art_painting", "cartoon"],
    6 :  ["sketch", "cartoon", "art_painting"],
    7 :  ["photo", "art_painting", "cartoon", "sketch"], # original
}

CONT_SEQUENCE_VLCS = {
    0 :  ["LabelMe", "SUN09", "VOC2007"], # original
    2 :  ["LabelMe", "VOC2007", "SUN09"],
    3 :  ["SUN09", "LabelMe", "VOC2007"],
    4 :  ["SUN09", "VOC2007", "LabelMe"],
    5 :  ["VOC2007", "LabelMe", "SUN09"],
    6 :  ["VOC2007", "SUN09", "LabelMe"],
    7 :  ["Caltech101", "LabelMe", "SUN09", "VOC2007"], # original
}

CONT_SEQUENCE_TINYIMAGENET = {
    0 :  ["gaussian_noise-5", "shot_noise-5", "impulse_noise-5", "defocus_blur-5", "glass_blur-5", "motion_blur-5", "zoom_blur-5", "snow-5", "frost-5", "fog-5", "brightness-5", "contrast-5", "elastic_transform-5", "pixelate-5", "jpeg_compression-5"],
    1 :  ['brightness-5', 'pixelate-5', 'gaussian_noise-5', 'motion_blur-5', 'zoom_blur-5', 'glass_blur-5', 'impulse_noise-5', 'jpeg_compression-5', 'defocus_blur-5', 'elastic_transform-5', 'shot_noise-5', 'frost-5', 'snow-5', 'fog-5', 'contrast-5'],
    2  : ['jpeg_compression-5', 'shot_noise-5', 'zoom_blur-5', 'frost-5', 'contrast-5', 'fog-5', 'defocus_blur-5', 'elastic_transform-5', 'gaussian_noise-5', 'brightness-5', 'glass_blur-5', 'impulse_noise-5', 'pixelate-5', 'snow-5', 'motion_blur-5'],
    3  : ['contrast-5', 'defocus_blur-5', 'gaussian_noise-5', 'shot_noise-5', 'snow-5', 'frost-5', 'glass_blur-5', 'zoom_blur-5', 'elastic_transform-5', 'jpeg_compression-5', 'pixelate-5', 'brightness-5', 'impulse_noise-5', 'motion_blur-5', 'fog-5'],
    4  : ['shot_noise-5', 'fog-5', 'glass_blur-5', 'pixelate-5', 'snow-5', 'elastic_transform-5', 'brightness-5', 'impulse_noise-5', 'defocus_blur-5', 'frost-5', 'contrast-5', 'gaussian_noise-5', 'motion_blur-5', 'jpeg_compression-5', 'zoom_blur-5'],
    5  : ['pixelate-5', 'glass_blur-5', 'zoom_blur-5', 'snow-5', 'fog-5', 'impulse_noise-5', 'brightness-5', 'motion_blur-5', 'frost-5', 'jpeg_compression-5', 'gaussian_noise-5', 'shot_noise-5', 'contrast-5', 'defocus_blur-5', 'elastic_transform-5'],
    6  : ['motion_blur-5', 'snow-5', 'fog-5', 'shot_noise-5', 'defocus_blur-5', 'contrast-5', 'zoom_blur-5', 'brightness-5', 'frost-5', 'elastic_transform-5', 'glass_blur-5', 'gaussian_noise-5', 'pixelate-5', 'jpeg_compression-5', 'impulse_noise-5'],
    7  : ['frost-5', 'impulse_noise-5', 'jpeg_compression-5', 'contrast-5', 'zoom_blur-5', 'glass_blur-5', 'pixelate-5', 'snow-5', 'defocus_blur-5', 'motion_blur-5', 'brightness-5', 'elastic_transform-5', 'shot_noise-5', 'fog-5', 'gaussian_noise-5'],
    8  : ['defocus_blur-5', 'motion_blur-5', 'zoom_blur-5', 'shot_noise-5', 'gaussian_noise-5', 'glass_blur-5', 'jpeg_compression-5', 'fog-5', 'contrast-5', 'pixelate-5', 'frost-5', 'snow-5', 'brightness-5', 'elastic_transform-5', 'impulse_noise-5'],
    9  : ['glass_blur-5', 'zoom_blur-5', 'impulse_noise-5', 'fog-5', 'snow-5', 'jpeg_compression-5', 'gaussian_noise-5', 'frost-5', 'shot_noise-5', 'brightness-5', 'contrast-5', 'motion_blur-5', 'pixelate-5', 'defocus_blur-5', 'elastic_transform-5'],
    10  : ['contrast-5', 'gaussian_noise-5', 'defocus_blur-5', 'zoom_blur-5', 'frost-5', 'glass_blur-5', 'jpeg_compression-5', 'fog-5', 'pixelate-5', 'elastic_transform-5', 'shot_noise-5', 'impulse_noise-5', 'snow-5', 'motion_blur-5', 'brightness-5'],
    
}

CONT_SEQUENCE_DOMAINNET126 = {
    0 :  ["clipart", "painting", "sketch"],
}


