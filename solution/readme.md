### SpaceNet5 Solution ###

We approached the SpaceNet5 road routing challenge as multiclass segmentation problem after observing the datset and geojsons
which had WKT Linestring for roads sorted as speedwise.The goal of the challenge is to get maximum APLS score. Thsi is was our
first experience with the SpaceNet dataset.

We backpropogated a lot of approaches and our submission scored a APLS score of around 0.21.

The SpaceNet5 solution involved the process of extracting huge dataset from S3 followed creating our own training masks with geojson
and many more steps in a month long journey which I'm planning to put up as a post shortly.Meanwhile, you can play with test dataset, model and have fun /mock my solution.
Kindly download required files from the drive link in notebook. Running this notebook in colab is advisable. Else make sure install all libraries in import section
