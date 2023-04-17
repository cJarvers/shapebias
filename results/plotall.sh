# Accuracy
python plot_accuracy.py -f ./accuracy/2023-03-09-20-28_classification_alexnet.pt -o ./figures/accuracy_alexnet.jpg --fdr  0.00625
python plot_accuracy.py -f ./accuracy/2023-03-09-20-37_classification_vgg19.pt -o ./figures/accuracy_vgg19.jpg --fdr  0.00625
python plot_accuracy.py -f ./accuracy/2023-03-13-12-09_classification_googlenet.pt -o ./figures/accuracy_googlenet.jpg --fdr  0.00625
python plot_accuracy.py -f ./accuracy/2023-03-09-20-43_classification_resnet50.pt -o ./figures/accuracy_resnet50.jpg --fdr  0.00625
python plot_accuracy.py -f ./accuracy/2023-03-09-21-01_classification_bagnet17.pt -o ./figures/accuracy_bagnet17.jpg --fdr  0.00625
python plot_accuracy.py -f ./accuracy/2023-03-09-21-42_classification_shape_resnet.pt -o ./figures/accuracy_shaperesnet.jpg --fdr  0.00625
python plot_accuracy.py -f ./accuracy/2023-03-09-21-51_classification_cornet.pt -o ./figures/accuracy_cornet.jpg --fdr  0.00625
python plot_accuracy.py -f ./accuracy/2023-03-09-21-34_classification_vit.pt -o ./figures/accuracy_vit.jpg --fdr  0.00625

# RSA results
python plot_rsa.py -f ./rsa/2023-03-13-11-44_rsa_image_alexnet_fixed.pt -o ./figures/rsa_image_alexnet.jpg --labelrotation 35 --fdr  0.00625
python plot_rsa.py -f ./rsa/2023-03-13-10-16_rsa_image_vgg19_fixed.pt -o ./figures/rsa_image_vgg19.jpg --labelrotation 35 --fdr  0.00625
python plot_rsa.py -f ./rsa/2023-03-13-12-43_rsa_image_googlenet_fixed.pt -o ./figures/rsa_image_googlenet.jpg --labelrotation 35 --fdr  0.00625
python plot_rsa.py -f ./rsa/2023-03-13-10-29_rsa_image_resnet50_fixed.pt -o ./figures/rsa_image_resnet50.jpg --labelrotation 35 --fdr  0.00625
python plot_rsa.py -f ./rsa/2023-03-13-10-30_rsa_image_bagnet17_fixed.pt -o ./figures/rsa_image_bagnet17.jpg --labelrotation 35 --fdr  0.00625
python plot_rsa.py -f ./rsa/2023-03-13-10-39_rsa_image_shape_resnet_fixed.pt -o ./figures/rsa_image_shaperesnet.jpg --labelrotation 35 --fdr  0.00625
python plot_rsa.py -f ./rsa/2023-03-13-10-54_rsa_image_cornet_fixed.pt -o ./figures/rsa_image_cornet.jpg --labelrotation 35 --fdr  0.00625
python plot_rsa.py -f ./rsa/2023-03-13-10-50_rsa_image_vit_fixed.pt -o ./figures/rsa_image_vit.jpg --labelrotation 35 --fdr  0.00625
