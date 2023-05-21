# Visually Impaired Assitive Tech (Object Detection and Recognition) 

Object-Detection is an assistive technology project aimed at helping visually impaired individuals by serving as artificial eyes. The project focuses on detecting and recognizing objects in front of the user and converting the recognized text into speech, thereby enhancing accessibility and independence for visually impaired individuals. 

## Objective

The primary objective of the Object-Detection project was to build a working prototype of a spectacle frame specifically designed for visually impaired individuals. The prototype needed to be free of any external wiring, and thus, the algorithm had to be processed on a Raspberry Pi module powered by a power bank. By leveraging object detection and recognition techniques, the project aimed to provide visually impaired individuals with the ability to recognize household items.

## Applied Technologies

The Object-Detection project utilized the following technologies:

- **Deep Learning:** Deep learning techniques were employed for object detection and recognition tasks. This involved training and deploying machine learning models capable of identifying and classifying objects.
- **Raspberry Pi:** The project utilized Raspberry Pi, a small and affordable computer, as the processing unit for the prototype. The Raspberry Pi module was powered by a portable power bank, making the prototype portable and easy to use.
- **GUI** - Tkinter: Tkinter, a Python library, was used to develop the graphical user interface (GUI) for the prototype. The GUI provided an intuitive and user-friendly interface for visually impaired individuals to interact with the system.
- **API and Flask:** APIs and Flask, a Python web framework, were employed to establish communication between the Raspberry Pi module and other components of the system. This allowed for seamless integration and interaction between different software components.
- **Object Detection and Recognition:** The project focused on implementing and optimizing object detection and recognition algorithms. These algorithms enabled the system to detect and recognize objects in the user's environment, providing valuable information to the visually impaired individual.

## Features

The Object-Detection project encompassed the following features:

- **Assistive Technology for Visually Impaired:** The project developed an assistive technology solution to aid visually impaired individuals. The system incorporated a camera module that captured the user's surroundings, and the object detection algorithm identified and recognized objects present in the captured images.
- **Text-to-Speech Conversion:** Once an object was detected and recognized, the system converted the recognized text into speech using a text-to-speech module. This allowed visually impaired individuals to hear the name of the object, providing them with valuable information about their environment.
- **Integration with Raspberry Pi-powered Eyeglasses:** The prototype was designed as a spectacle frame equipped with a Raspberry Pi module and a small camera. The Raspberry Pi-powered eyeglasses were lightweight and portable, making them convenient for daily use by visually impaired individuals.
- **Machine Learning Model Training and Testing:** To achieve accurate object detection and recognition, machine learning models were trained and tested. Techniques such as transfer learning and data augmentation were employed to improve the performance of the models. The models were optimized to achieve a high accuracy rate, ensuring reliable object detection and recognition.

## Files

The repository includes the following files:

- *Object Detection/:* This directory contains the code for object detection, including the implementation of the object detection algorithm and the necessary dependencies.
- *Object-Recognition/:* This directory contains the uploaded model file for object recognition. The trained model is used to recognize objects detected in the captured images.
- **README.md:** This README file providing a detailed overview of the Object-Detection project.
- Various **.pkl** files: These files contain the dataset used for training the machine learning models. They are essential for retraining or further optimizing the models.




