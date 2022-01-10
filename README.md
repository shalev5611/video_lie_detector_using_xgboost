# video_lie_detector_using_xgboost
a video lie detector using OpenFace and xgboost
for this project, I used the OpenFace tool on a docker container.
this project contains 2 scripts:
train_xgboost_model.py- trains the model on the action units data
predict_with_xgboost.py generates predictions from video files

in this project, I use OpenFace to extract the action units, an encoding method for facial movements, and passing a list of average of action units of each movement.
the model currently has about 70% percent accuracy, with a 50/50 label distribution in the dataset. that means the model is capable of learning the dataset,
and could improve if there was more data available

datset size: 121 samples
distribution: 61 lies, 60 truthful
accuracy score 0.71
precision score 0.73
recall score 0.67
f1 score 0.7

