from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import cv2
from EYGDS3Website.settings import ORB, INTERPRETER, CATEGORY_LABELS, evaluateClassifier, evaluateRNN, extract_keywords, finalKeywords,chi2_distance
from .models import Image
from datetime import datetime
import fitz
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import collections
from EYGDS3Website.settings import COLOR_DESCRIPTOR

# Create your views here.

@csrf_exempt
def homePage(request):

    if(request.method == 'GET'):
        return render(request,"NavBar.html",{})
    
    else:
        if("image_upload" in request.POST):
            myImage = request.FILES['myimage']
            extension = myImage.name.split(".")[-1]
            folder = 'static/Media'
            fs = FileSystemStorage(location=folder)
            now = datetime.now()
            date_time = now.strftime("%m%d%Y%H%M%S")
            date_time = date_time+"."+extension


            fs.save(date_time,myImage)

            print("Recieved image.....")

            image_path = folder+"/"+date_time

            img = cv2.imread(image_path)

            # FEATURE EXTRACTION    
            img1 = cv2.resize(img, (300, 300), cv2.INTER_AREA)
            features_from_image = COLOR_DESCRIPTOR.describe(img1)

            # orb = cv2.xfeatures2d.SURF_create()
            img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

            kp1, des1 = ORB.detectAndCompute(img1, None)



            # OBJECT DETECTOR

            input_details = INTERPRETER.get_input_details()
            output_details = INTERPRETER.get_output_details()

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (300, 300), cv2.INTER_AREA)
            img_rgb = img_rgb.reshape([1, 300, 300, 3])

            INTERPRETER.set_tensor(input_details[0]['index'], img_rgb)
            INTERPRETER.invoke()

            det_classes = INTERPRETER.get_tensor(output_details[1]['index'])[0]
            
            det_classes = list(set(det_classes))

            list_of_words = []

            for i in det_classes:
                list_of_words.append(CATEGORY_LABELS[str(int(i))]['name'])

            print(list_of_words)


            # CNN-RNN

            result, attention_plot = evaluateRNN(image_path)
            result = ' '.join(result[:-1])
            print(result)

            verbs = extract_keywords(result)
            print(verbs)

            list_of_words.extend(verbs)

            # finalCaption = result+" "+list_of_words
            # print('Prediction Caption:', finalCaption)

            #CNN CLASSIFIER

            output = evaluateClassifier(image_path)
            print(output)
            list_of_words.extend(output)

            list_of_words = list(set(list_of_words))

            print(list_of_words)

            des1 = np.array(des1)

            image_path = folder+"/"+date_time

            features_from_image = np.array(features_from_image)

            request.session['des'] = des1.tolist()
            request.session['kp_len'] = len(kp1)
            request.session['features'] = features_from_image.tolist()

            request.session['path'] = image_path
            request.session['tags'] = list_of_words
            request.session['name'] = date_time

            return redirect('/tags')

        # PDF PROCESSING

        elif ("pdf_upload" in request.POST):
            myPdf = request.FILES['pdfFile']
            extension = myPdf.name.split(".")[-1]
            folder = 'static/Media'
            fs = FileSystemStorage(location=folder)
            now = datetime.now()
            date_time = now.strftime("%m%d%Y%H%M%S")
            date_time = date_time+"."+extension


            fs.save(date_time,myPdf)

            print("Recieved pdf.....")

            pdf_path = folder+"/"+date_time

            doc = fitz.open(pdf_path)
            counter = 0
            for i in range(len(doc)):
                for img in doc.getPageImageList(i):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n < 5: 

                        now = datetime.now()
                        temp_time = now.strftime("%m%d%Y%H%M%S")
                        img_file_name = temp_time+str(counter)+".jpg"
                        save_path = folder+"/"+img_file_name
                        pix.pil_save(save_path)
                        

                        temp_img = cv2.imread(save_path)
                        img_temp = cv2.resize(temp_img, (300, 300), cv2.INTER_AREA)
                        features_from_image = COLOR_DESCRIPTOR.describe(img_temp)
                        img_temp = cv2.cvtColor(img_temp,cv2.COLOR_BGR2GRAY)

                        kp1, des1 = ORB.detectAndCompute(img_temp, None)

                        data = {}
                        data['des'] = des1
                        data['kp_length'] = len(kp1)
                        data['features'] = features_from_image
                        myObj = Image(name=img_file_name,data = data,pdf_file=pdf_path)
                        counter+=1
                        myObj.save()

                    pix = None

            return render(request,"NavBar.html",{})

        # VIDEO PROCESSING FEATURE EXTRACTOR

        elif ("video_upload" in request.POST):

            myVideo = request.FILES['videoFile']
            extension = myVideo.name.split(".")[-1]
            folder = 'static/Media'
            fs = FileSystemStorage(location=folder)
            now = datetime.now()
            date_time = now.strftime("%m%d%Y%H%M%S")
            date_time = date_time+"."+extension


            fs.save(date_time,myVideo)

            print("Recieved video.....")

            video_path = folder+"/"+date_time

            frame_list = []

            feature_list = []


            cap = cv2.VideoCapture(video_path)
            processed = 0
            while True:
                ret,frame = cap.read()

                if(processed%10==0):

                    if ret == True:
                        gray = cv2.resize(frame, (300, 300), cv2.INTER_AREA)

                        features_from_image = COLOR_DESCRIPTOR.describe(gray)
                        feature_list.append(features_from_image)


                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame_list.append(gray)



                        processed+=1

                    else:
                        break
                else:
                    processed+=1
            cap.release()

            for i in range(len(frame_list)):
                img_temp = frame_list[i]
                now = datetime.now()
                temp_time = now.strftime("%m%d%Y%H%M%S")
                img_file_name = temp_time+str(i)+".jpg"
                save_path = folder+"/"+img_file_name

                cv2.imwrite(save_path,img_temp)

                # ADD COLOR EXTRACTION

                kp1, des1 = ORB.detectAndCompute(img_temp, None)

                features = feature_list[i]


                data = {}
                data['des'] = des1
                data['kp_length'] = len(kp1)
                data['features'] = features
                myObj = Image(name=img_file_name,data = data,video_file=video_path)
                myObj.save()

            return render(request,"NavBar.html",{})

        elif ("searchButton" in request.POST):
            text = request.POST['searchText']
            print(text)

            if("search_sentence" in request.POST):
                list_of_words = extract_keywords(text)

                print(list_of_words)


                all_images = Image.objects.filter(tagged=True)


                print("HERE")

                matcherDict = {}

                for i in range(len(all_images)):
                    l2 = all_images[i].tags
                    outputs = list(set(list_of_words).intersection(l2))
                    print(outputs)
                    length = len(outputs)
                    if(length>0):
                        matcherDict[i] = length

                print(matcherDict)

                data = dict(sorted(matcherDict.items(), key=lambda item: item[1], reverse=True))
                
                list_of_paths = []
                for k in data:
                    print(all_images[k].image_file)
                    list_of_paths.append(all_images[k].image_file)

                context = {
                    'imagePaths':list_of_paths
                }

                return render(request,"img-gallery.html",context)

            if("search_tags" in request.POST):
                list_of_words = text.split(", ")

                print(list_of_words)


                all_images = Image.objects.filter(tagged=True)


                print("HERE")

                matcherDict = {}

                for i in range(len(all_images)):
                    l2 = all_images[i].tags
                    outputs = list(set(list_of_words).intersection(l2))
                    print(outputs)
                    length = len(outputs)
                    if(length>0):
                        matcherDict[i] = length

                print(matcherDict)

                data = dict(sorted(matcherDict.items(), key=lambda item: item[1], reverse=True))
                
                list_of_paths = []
                for k in data:
                    print(all_images[k].image_file)
                    list_of_paths.append(all_images[k].image_file)

                context = {
                    'imagePaths':list_of_paths
                }

                return render(request,"img-gallery.html",context)




        elif("ris_upload" in request.POST):

            myReverseImage = request.FILES['myRIS']
            extension = myReverseImage.name.split(".")[-1]
            folder = 'static/Media'
            fs = FileSystemStorage(location=folder)
            now = datetime.now()
            date_time = now.strftime("%m%d%Y%H%M%S")
            date_time = date_time+"."+extension


            fs.save(date_time,myReverseImage)

            print("Recieved RIS......")

            reverseImagePath = folder+"/"+date_time
            img = cv2.imread(reverseImagePath)
            img1 = cv2.resize(img, (300, 300), cv2.INTER_AREA)

            queryFeature = COLOR_DESCRIPTOR.describe(img1)

            img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

            kp1, des1 = ORB.detectAndCompute(img1, None)       

            all_images = Image.objects.all()
            bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)    
            
            kp_length = len(kp1)

            matcherDict = {}
            counter = 0

            print(type(des1))

            for i in range(len(all_images)):

                tempData = {}

                data = all_images[i].data
                des = data['des']
                des = np.array(des)

                featuresOfImage = data['features']
                featuresOfImage = np.array(featuresOfImage)

                d = chi2_distance(featuresOfImage, queryFeature)
                
                tempData['feature_score'] = d

                print(type(des))


            # MATCHING ALGORITHM

                matches = bf.knnMatch(np.asarray(des1,np.float32),np.asarray(des,np.float32),k=2)
                good = []

                print(type(matches))

                for m,n in matches:
                    if m.distance < 0.85*n.distance:
                        good.append([m])
                        a=len(good)
                        percent=(a*100)/kp_length
                        if percent>=20.00:
                            break
                if(percent>2):
                        tempData['SURF_score'] = percent

                print(i)

                matcherDict[i] = tempData

            print(matcherDict)
            list_of_paths = []

            for keys in matcherDict.keys():
                feature_score = float(matcherDict[keys]['feature_score'])
                surf_score = float(matcherDict[keys]['SURF_score'])

                

                if(feature_score<7 or surf_score>17.5):
                    path_of_image = all_images[int(keys)].name
                    print(all_images[int(keys)].name)
                    path_of_image = "static/Media/"+path_of_image
                    list_of_paths.append(path_of_image)

            # if(len(list_of_paths)>0):
            
            print(list_of_paths)
            context = {
                    'imagePaths':list_of_paths
                }
            return render(request,"img-gallery.html",context)
            # else:
            #     return redirect('/')
                




        else:
            pass

@csrf_exempt
def galleryPage(request):

    if(request.method == "GET"):
        
        all_images = Image.objects.all()
        list_of_paths = []
        for i in range(len(all_images)):
            tempPath = all_images[int(i)].name
            tempPath = 'static/Media/'+tempPath
            list_of_paths.append(tempPath)

        context = {
            'imagePaths':list_of_paths
        }

        request.session['paths'] = list_of_paths

        print(list_of_paths)

        return render(request,"img-gallery2.html",context)

    elif ("searchButton" in request.POST):
        text = request.POST['searchText']
        print(text)

        list_of_words = text.split(", ")

        print(list_of_words)


        all_images = Image.objects.filter(tagged=True)


        print("HERE")

        matcherDict = {}

        for i in range(len(all_images)):
            l2 = all_images[i].tags
            outputs = list(set(list_of_words).intersection(l2))
            print(outputs)
            length = len(outputs)
            if(length>0):
                matcherDict[i] = length
                #hello

        print(matcherDict)

        data = dict(sorted(matcherDict.items(), key=lambda item: item[1], reverse=True))
        
        list_of_paths = []
        for k in data:
            print(all_images[k].image_file)
            list_of_paths.append(all_images[k].image_file)

        context = {
            'imagePaths':list_of_paths
        }

        return render(request,"img-gallery.html",context)

    else:
        list_of_paths = request.session['paths']

        for i in list_of_paths:
            if i in request.POST:
                print(i)
                target_image = Image.objects.filter(image_file=i)[0]
                print(target_image.tags)

                request.session['des'] = target_image.data['des']
                request.session['kp_len'] = target_image.data['kp_length']
                request.session['features'] = target_image.data['features']

                request.session['path'] = target_image.image_file
                request.session['tags'] = target_image.tags
                request.session['name'] = target_image.name

                target_image.delete()

                return redirect('/tags')




@csrf_exempt
def keywordPage(request):

    if(request.method=="GET"):
        print("GET HERE")

        tags = request.session['tags']
        path = request.session['path']

        print(tags)

        context = {
            'tags':tags,
            'path':path,
        }
        return render(request,"Keywords.html",context)

    else:

        if ("responses" in request.POST):
            tags = request.session['tags']
            finalList = []
            for i in tags:
                if(i in request.POST):
                    finalList.append(i)

            if("additional_tags" in request.POST):
                string = request.POST.get("additional_tags")
                l = string.split(", ")
                if(l[0]!=''):
                    finalList.extend(l)

            print(finalList)

            keyWords = finalKeywords(finalList,1)
            print(keyWords)

            keyWords = list(set(keyWords))

            request.session['tags'] = keyWords

        return redirect('/finalise')

@csrf_exempt
def finalisePage(request):

    if(request.method=="GET"):
        print("GET HERE")

        tags = request.session['tags']
        path = request.session['path']

        print(tags)

        context = {
            'tags':tags,
            'path':path,
        }
        return render(request,"submit.html",context)

    else:

        if("done" in request.POST):

            #Background task for even more keywords generation

            #DATA STORAGE



            des = request.session['des']
            kp_len = request.session['kp_len']
            features_from_image = request.session['features']
            nameOfImage = request.session['name']
            imagePath = request.session['path']
            tags = request.session['tags']

            tags = list(set(tags))

            print("DONE SECTION")
            print(tags)

            # furtherExpandKeywords(tags,des,kp_len,nameOfImage,imagePath)

            # printHello()


            data = {}
            data['des'] = np.array(des)
            data['kp_length'] = kp_len
            data['features'] = features_from_image

            

            myObj = Image(name=nameOfImage,data = data,image_file=imagePath,tagged = True, tags = tags)
            myObj.save()

            return redirect('/')

        else:

            return redirect('/tags')


# @background(schedule=1)
# def furtherExpandKeywords(word_list,des,kp_length,nameOfImage,imagePath):
#     keyWords = finalKeywords(word_list,5)
#     data = {}
#     data['des'] = np.array(des)
#     data['kp_length'] = kp_length

#     tags = list(set(keyWords))

#     print("BACKGROUND TASK")
#     print(tags)

#     myObj = Image(name=nameOfImage,data = data,image_file=imagePath,tagged = True, tags = tags)
#     myObj.save()

# @background(schedule=1)
# def printHello():
#     print("HELLO WORLD")



