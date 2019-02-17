import sys
import torch.nn as nn
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_densenet(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    model_params_optimized = None
    input_size

    if model_name == 'densenet121':
        model_ft = models.densenet121(pretrained=use_pretrained)
    elif model_name == 'densenet161':
        model_ft = models.densenet161(pretrained=use_pretrained)
    elif model_name == 'densenet169':
        model_ft = models.densenet169(pretrained=use_pretrained)
    elif model_name == 'densenet201':
        model_ft = models.densenet201(pretrained=use_pretrained)
    else:
        print('{} is not a valid DenseNet in torchvision'.format(model_name))
        sys.exit(1)

    set_parameter_requires_grad(model_ft, feature_extract)
    num_features = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_features, num_classes)
    if feature_extract:
        model_params_optimized = model_ft.classifier.parameters()
    else:
        model_params_optimized = model_ft.parameters()
    input_size = 224

    return model_ft, model_params_optimized, input_size

def initialize_resnet(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    model_params_optimized = None
    input_size = 0

    if model_name == 'resnet18':
        model_ft = models.resnet18(pretrained=use_pretrained)
    elif model_name == 'resnet34':
        model_ft = models.resnet34(pretrained=use_pretrained)
    elif model_name == 'resnet50':
        model_ft = models.resnet50(pretrained=use_pretrained)
    elif model_name == 'resnet101':
        model_ft = models.resnet101(pretrained=use_pretrained)
    elif model_name == 'resnet152':
        model_ft = models.resnet152(pretrained=use_pretrained)
    else:
        print('{} is not a valide ResNet in torchvision'.format(model_name))
        sys.exit(1)

    set_parameter_requires_grad(model_ft, feature_extract)
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features, num_classes)
    if feature_extract:
        model_params_optimized = model_ft.fc.parameters()
    else:
        model_params_optimized = model_ft.parameters()
    input_size = 224

    return model_ft, model_params_optimized, input_size

def initialize_vgg(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    model_params_optimized = None
    input_size = 0

    if model_name == 'vgg11':
        model_ft = models.vgg11(pretrained=use_pretrained)
    elif model_name == 'vgg11_bn':
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
    elif model_name == 'vgg13':
        model_ft = models.vgg13(pretrained=use_pretrained)
    elif model_name == 'vgg13_bn':
        model_ft = models.vgg13_bn(pretrained=use_pretrained)
    elif model_name == 'vgg16':
        model_ft = models.vgg16(pretrained=use_pretrained)
    elif model_name == 'vgg16_bn':
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
    elif model_name == 'vgg19':
        model_ft = models.vgg19(pretrained=use_pretrained)
    elif model_name == 'vgg19_bn':
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
    else:
        print('{} is not a valid vgg model in torchvision'.format(model_name))
        sys.exit(1)

    set_parameter_requires_grad(model_ft, feature_extract)
    num_features = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_features,num_classes)
    if feature_extract:
        model_params_optimized = model_ft.classifier[6].parameters()
    else:
        model_params_optimized = model_ft.parameters()
    input_size = 224

    return model_ft, model_params_optimized, input_size

def initialize_squeezenet(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    model_params_optimized = None
    input_size = 0

    if model_name == 'squeezenet1_0':
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
    elif model_name == 'squeezenet1_1':
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
    else:
        print('{} is not a valid squeeze net in torchvision'.format(model_name))
        sys.exit(1)

    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    if feature_extract:
        model_params_optimized = model_ft.classifier[1].paramters()
    else:
        model_params_optimized = model_ft.parameters()
    input_size = 224

    return model_ft, model_params_optimized, input_size

# main method
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    model_params_optimized = None
    input_size = 0

    if 'densenet' in model_name:
        model_ft, model_params_optimized, input_size = initialize_densenet(model_name, num_classes, feature_extract, use_pretrained)
    elif 'resnet' in model_name:
        model_ft, model_params_optimized, input_size = initialize_resnet(model_name, num_classes, feature_extract, use_pretrained)
    elif 'vgg' in model_name:
        model_ft, model_params_optimized, input_size = initialize_vgg(model_name, num_classes, feature_extract, use_pretrained)
    elif 'squeezenet' in model_name:
        model_ft, model_params_optimized, input_size = initialize_squeezenet(model_name, num_classes, feature_extract, use_pretrained)
    elif model_name == 'alexnet':
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_features, num_classes)
        if feature_extract:
            model_params_optimized = model_ft.classifier[6].parameters()
        else:
            model_params_optimized = model_ft.parameters()
        input_size = 224
    #elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        #model_ft = models.inception_v3(pretrained=use_pretrained)
        #set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        #num_features = model_ft.AuxLogits.fc.in_features
        #model_ft.AuxLogits.fc = nn.Linear(num_features, num_classes)
        # Handle the primary net
        #num_features = model_ft.fc.in_features
        #model_ft.fc = nn.Linear(num_features,num_classes)
        #input_size = 299
    else:
        print('{} is not a valid model in torchvision'.format(model_name))
        sys.exit(1)

    return model_ft, model_params_optimized, input_size
