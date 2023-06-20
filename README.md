First, by running create_npy.py file,
we create .pt file for the model with the name 'kr16-conv2-lr0.1-batch.pt' under checkpoints folder. 

It also create 'estimations_test.npy' file for test dataset which has size 2000, which has images directory under test folder, but takes 100 of them. If you want to change the size, you can edit `IMAGE_NUMBER` in create.npy file and run it. 


After running

`python evaluate.py estimations_test.npy test_images.txt`

command, you can get the average accuracy for test dataset.


Since the order of the predictions is random, we provide test_images.txt file which has the file names with the same order. 


If you need to create .npy file again, remove the test_images.txt file before as follows:


`rm test_images.txt`

`python create_npy.py`

`python evaluate.py estimations_test.npy test_images.txt`


If code does not run, you can use requirements.txt