### SpaceNet5 Solution ###

The SpaceNet5 road routing challenge was approached as a multiclass segmentation problem after observing the dataset and geojsons which had WKT Linestring for roads sorted as speedwise.The goal of the challenge is to get maximum APLS score. This is was the first experience with the SpaceNet dataset.

Backpropagated a lot of approaches and submission scored a APLS score of around 0.21.

The SpaceNet5 solution involved the process of extracting huge dataset from S3 followed creating our own training masks with geojson and many more steps in a month long journey which I'm planning to put up as a post shortly.Meanwhile, you can play with test dataset, model and have fun/mock my solution.
Kindly download required files from the drive link in notebook. Running this notebook in colab is advisable. Else make sure all libraries in import section are installed. Stay Tuned !!!
