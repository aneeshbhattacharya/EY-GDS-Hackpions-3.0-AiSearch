# EY-GDS-Hackpions-3.0-AiSearch
This is the Django project which we developed for the Ernst and Young GDS Hackpions 3.0 and won 1st place and cash prize of 1.5 Lakh INR. <br>
**Problem Statement**: Image Search and Analysis <br>
**Team Name**: AiSearch <br>
**Competition Result**: https://www.hackerearth.com/challenges/hackathon/ey-gds-hackpions-30/

## Main Page
![1](https://user-images.githubusercontent.com/68210639/180231043-3793d05a-a13f-4802-a92b-d802cfb6593c.png)
## How it works?
<ul>
  <li>Uploading and Tagging System: (for images, PDFs and videos)</li>
   
   <ul>
      <li>User can upload any of the three file types - images, videos and pdf files.
</li>
      <li>The uploaded image files are processed and automatically tagged in a pipeline using CNN-RNN, CNN classifier and Object detection models (which have been trained using the ImageNet and COCO 2017 datasets and can tag over 1100 classes).  The generated tags are further processed in an NLP pipeline which generate similar words hence giving the user a large number of tags and they can then choose the most appropriate tags from them and further add their own tags.
</li>
      <li>The image frames are first extracted from pdf and video files and are then processed using SURF and Color Histogram feature extraction. These features are extracted and saved to be used for reverse image searches
</li>
   </ul>
  
  <li>Search and Retrieval System: (using sentences, tags and reverse images search)</li>
   <ul>
      <li>User can either search by keywords or sentences OR can upload an images as a query and get related images as the output.</li>
      <li>Keywords/ Phrases or Sentences- User searches in natural English language and NLP is used to extract and generate keywords. The generated tags are matched with tags of images in the database and the most appropriate images are displayed.</li>
      <li>Reverse Image Search: Image uploaded by the user is processed using SURF and Color Histogram feature extractors and matched with features of images, PDFs and videos in the database and displayed.</li>
   </ul>
</ul>

## Setup:
The following has been tested on Ubuntu 20.04.
### Create Conda Environment:
```
conda create -n EY python=3.7.11
conda activte EY
```
### Install Tesseract OCR
```
sudo apt-get install tesseract-ocr
```
### Clone and install requirements:
```
git clone https://github.com/aneeshbhattacharya/EY-GDS-Hackpions-3.0.git
cd EY-GDS-Hackpions-3.0
pip install -r requirements.txt
```
### Download ModelFiles:
```
https://drive.google.com/drive/folders/11hDtSMueVsk_njiuSESL-gI30MrpmD2Y?usp=sharing
```
Please place ModelFiles inside EY-GDS-Hackpions-3.0/static/
### Deploy on localhost and run:
```
python manage.py runserver
```
Open 127.0.0.1:8000 on your web browser and follow instructions on the website
