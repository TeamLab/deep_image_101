### Keras collection 

#### Requirment
* Keras, Scikit-learn, Numpy

#### Function Info

1. img_utils.py
    - Usage 
    ~~~
    histogram_equalizer(img)
    
    load_img(path)
    
    save_img(img, path)
    
    resize_img(img, img_h, img_w, method='bilinear')
    
    minmax_img(img)
    
    mean_std_img(imgs, method='sample')
    
    auroc_plot(y_true, y_pred, path="auroc_plot.png")
    
    ~~~

2. data_utils.py
    
    -Usage
    ~~~
    k_fold_split(x_data, y_data, k=5)
    
    split_dataset(x_data, y_data, size=0.2, random_state=42)
    
    label_smoothing(y_data, epsilon=0.01)
    ~~~

3. metrics_utils.py

    -Usage
    ~~~
    kl_divergence(y_true, y_pred)
    
    dice_coefficient(x, y, smooth=1)
    
    auroc_score(y_true, y_pred)
    ~~~

4. loss_function.py

    -Usage
    ~~~
    segmetation_binary_loss(y_true, y_pred)
    
    segmentation_dice_loss(y_true, y_pred)
    ~~~

5. keras_model

    5.1) Segmentation
        
    -Unet 
    ~~~
    from segmentation_model import *
    
    unet(img_h, img_w, channel, num_filter=64, kernel_size=3, init='he_normal')
    ~~~
    
    5.2) Classification
    
    -Xception
    ~~~
    from classification_model import *
    
    xception(img_h, img_w, channel, classes, gmp=True, gap=True, fc=False, summary=True)
    ~~~
    
    -Inception-v3
    ~~~
    from classification_model import *
    
    inception_v3(img_h, img_w, channel, classes, gmp=True, gap=True, fc=False, summary=True)
    ~~~
    
    5.3) Regression
    
    -Base regression
    ~~~
    from regression_model import *
    
    baseline_regression(img_h, img_w, channel, gmp=True, gap=True, summary=True)
    ~~~
    
    5.4) Detection
 
6. Visualize

    6.1 CAM (Class Activation Map)
    
    -Usage
    ~~~
    from visualizer import *
    
    cam(model, img_dir, out_dir, img_normalize=True, histogram_equalizer=True)
    ~~~
    
    6.2 Grad-CAM (Gradient Class Activation Map)
    
    
    
#### References