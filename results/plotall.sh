# Accuracy
python plot_accuracy.py -f ./accuracy/2022-11-29-22-26_classification_vgg19.pt -o ./figures/accuracy_vgg19.png --show --fdr 0.0083
python plot_accuracy.py -f ./accuracy/2022-11-29-22-31_classification_resnet50.pt -o ./figures/accuracy_resnet50.png --show --fdr 0.0083
python plot_accuracy.py -f ./accuracy/2022-11-30-09-04_classification_bagnet17.pt -o ./figures/accuracy_bagnet17.png --show --fdr 0.0083
python plot_accuracy.py -f ./accuracy/2022-11-30-09-08_classification_shape_resnet.pt -o ./figures/accuracy_shaperesnet.png --show --fdr 0.0083
python plot_accuracy.py -f ./accuracy/2022-11-30-09-14_classification_cornet.pt -o ./figures/accuracy_cornet.png --show --fdr 0.0083
python plot_accuracy.py -f ./accuracy/2022-11-30-09-19_classification_vit.pt -o ./figures/accuracy_vit.png --show --fdr 0.0083

# RSA results
python plot_rsa.py -f ./rsa/2022-11-29-16-16_rsa_image_cornet_fixed.pt -o ./figures/rsa_image_cornet.png --labelrotation 35 --fdr 0.0083
python plot_rsa.py -f ./rsa/2022-11-29-16-03_rsa_image_bagnet17_fixed.pt -o ./figures/rsa_image_bagnet17.png --labelrotation 35 --fdr 0.0083
python plot_rsa.py -f ./rsa/2022-11-29-15-50_rsa_image_shape_resnet_fixed.pt -o ./figures/rsa_image_shaperesnet.png --labelrotation 35 --fdr 0.0083
python plot_rsa.py -f ./rsa/2022-11-29-15-45_rsa_image_resnet50_fixed.pt -o ./figures/rsa_image_resnet50.png --labelrotation 35 --fdr 0.0083
python plot_rsa.py -f ./rsa/2022-11-29-16-29_rsa_image_vit_fixed.pt -o ./figures/rsa_image_vit.png --labelrotation 35 --fdr 0.0083
python plot_rsa.py -f ./rsa/2022-11-29-15-49_rsa_image_vgg19_fixed.pt -o ./figures/rsa_image_vgg19.png --labelrotation 35 --fdr 0.0083