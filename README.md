## MyNaturewatch Convolutional Neural Network

# Problem
The [MyNaturewatch DIY camera](https://mynaturewatch.net/daylight-camera-instructions) is a great DIY birding/outdoors camera to take pictures of wildlife automatically. It uses a Raspberry Pi Zero and a Pi Zero camera + sensor to detect when there is movement and takes a picture of it. The problem is, more often than not, the camera takes pictures of anything that moves like leaves. I find myself scrolling through an endless gallery of nothing and there are very few pictures of actual animals. This network solves that problem.

![no animal](/resources/2020-05-14-14-38-41.jpg)

vs

![yes animal](/resources/2020-05-14-08-31-45.jpg)

# What will this program do? Goals?
- Download all the photos from the Naturewatch Camera (RasPi)
- Classify them
- Make directories based on the dates of the animal pictures in the ~/Pictures folder

- Goal: 95%+ testing accuracy

# Dataset
The dataset contains 372 pictures of animals and 1,934 pictures of other things (no animals). I prefer not to upload the dataset because it includes human faces of people I know (very random things to add more variety).

There is an obvious bias in the dataset; the non-animal pictures outnumber the animal pictures 5 to 1. This teaches the model that there are more non-animal pictures and that animal pictures are rarer. 

# Todo
- Use SSH to download photos off the NatureWatch camera
- expand the dataset, more variety of images. More brown/gray animals like pigeons and squirrels

# Disclaimer
I am not part of the MyNaturewatch team. 