EUDA (Emotion Understanding and Development AI)

## Inspiration
The inspiration behind Emotion Understanding and Development AI (EUDA) was to help people improve their emotional well-being by leveraging the power of advanced facial and vocal recognition technology. We wanted to create a bot that could act as a personalized resource to help people manage difficult emotions effectively. People often feel uncomfortable talking to others about their emotions, particularly if they fear judgment, misunderstanding, or rejection. In these cases, our AI chatbot (EUDA) can help individuals feel more comfortable expressing their emotions by offering a safe and private space for them to do so. Our vision for EUDA is to meaningfully impact communities by eliminating needless mental-health-related deaths and increasing overall mental health, which aligns with the theme of the hackathon. Mental health disorders like anxiety and depression cause major gaps in a community, which EUDA can help bridge.  We recognize that emotional well-being is a major concern that affects individuals from everywhere, and our website aims to promote emotional well-being by helping people navigate difficult emotions and improve their overall emotional health.

## What it does
EUDA is an AI-powered platform that uses advanced facial and vocal recognition technology, sentiment analysis, and machine learning to detect changes in emotions and provide personalized resources and support to individuals dealing with difficult emotions. We have three main models, a face recognition model using computer vision, a speech emotion recognition model that has been trained using machine learning, and a sentiment analysis model. Our facial recognition technology uses computer vision to analyze the expressions of the user to determine the emotions they are feeling. Our sentiment analysis technology examines a user’s voice to determine what they are saying, then uses natural language processing to analyze their sentiment. Our vocal recognition technology identifies the features of a user’s voice to determine their emotional state. EUDA intelligently uses all of the data from these models to offer personalized guidance, self-care tips, mindfulness exercises, cognitive behavioral therapy techniques, and other evidence-based interventions that are tailored to the unique needs of the individual in the form of an emotionally aware AI chatbot. EUDA is designed to provide a safe and private space for individuals to express their emotions without fear of negative consequences. It is user-friendly and accessible to everyone, regardless of their technical proficiency. Overall, EUDA is a comprehensive emotional support platform that aims to improve the emotional well-being of individuals and communities by providing sustainable, long-term solutions in the form of an AI chatbot.

## How we built it
EUDA was built using several languages such as Python, HTML, CSS, and Javascript, and required the usage of cutting-edge technologies such as the Flask web framework library, OpenCV, OpenAI, TensorFlow, Keras, and NTLK. We leveraged advanced facial and vocal recognition paired with sentiment analysis technology to detect changes in emotions. For our facial emotional recognition machine learning model, we utilized computer vision through the OpenCV library to analyze facial expressions and detect the emotions and emotional changes of the user. For our vocal recognition machine learning model, we trained our own model using Keras and Tensorflow with around 10,000 audio files from several datasets such as RAVDESS, TESS, SAVEE, and CREMA-D. We preprocessed our input data and added 8 augmentations to them to increase the robustness of our model. For our sentiment analysis model, we process audio input from the user using the Speech Recognition library to turn it into AI readable text, then process the sentiment of the text through the NLTK library to generate a sentiment and emotionally aware response. To make our website secure, we used the SQLite database to make an authentication system and then encrypted it by using Argon2. Our website provides a panel page where users are able to access the facial emotional recognition model and this page also confirms that they are visible to their camera to ensure the proper functionality of our facial emotional recognition model. The panel is updated in real-time using AJAX, Flask, JS, and HTML, and boundary boxes are displayed around targets that have detected emotions. After the user’s face is centered and detected on the panel page, the user can navigate to our AI chatbot, EUDA using the “Next” button. The chat interface was built with HTML, CSS, and JavaScript and uses the Flask web framework to handle the backend of the application and AJAX to handle the communication between the front and back end. HTML was used for the general structure of the page and CSS was used to style the various elements such as the chat box, message bubbles, and avatar. For the message input and submission, we used a combination of HTML and JavaScript to create a countdown timer and submit the user's message to the backend when the timer reaches zero. 

## Challenges we ran into
We faced several significant challenges in this hackathon. One of the challenges we faced was integrating all of the different technologies seamlessly. We overcame this challenge by following a modular approach, where we developed each component of EUDA separately, and then integrated it all into the website. We also had to perform countless tests of our product to make sure all of the models functioned together. Another challenge was coming up with an authentication system to secure each chat and ensure the privacy of the users. None of our team members had much experience with authentication so we had to conduct a lot of research to progress. Another one of the greatest challenges that we faced was obtaining and preparing a high-quality dataset and training our model. We needed diverse datasets that included a wide range of emotions expressed in different tones and pitches to ensure high accuracy. The preprocessing of the data was also a difficult task which involved extracting relevant features from the audio files and preparing them for input into our machine-learning model. This required a lot of experimentation with different techniques and parameters to find the optimal approach. Furthermore, we required significant computational power and time to train our speech-emotional recognition model. Another one of our challenges was putting a bounding box on the face of the users to accurately recognize their emotions of the users. But we were also able to overcome this through comprehensive research. We also originally thought to make this as an app, but due to time constraints and the amount of time it took to train the models, we decided to make a website instead. 


## Accomplishments that we're proud of
There are many accomplishments that we are proud of in developing EUDA. First of all, we are proud of our team's collaboration and dedication to bringing this project to life. Each team member contributed their unique skills and expertise to create a functional and innovative product. It was our first time working with machine learning and it was really painful to make the model work with basically no prior knowledge. We’re really proud that we actually managed to train the model in speech emotion recognition. We are also proud of our chatbot's ability to accurately detect and respond to the emotions of the user. This was a challenging aspect of the project, as emotions can be complex and difficult to interpret. However, through the use of natural language processing and machine learning techniques, we were able to develop a chatbot that can accurately detect and respond to emotions in real time. Finally, we are proud of the potential impact that EUDA could have on mental health support. By providing an accessible and responsive chatbot that can understand and respond to the emotions of the user, we hope to provide a valuable resource for those in need of emotional support. We believe that EUDA has the potential to make a positive difference in the lives of many people as it can help lead users away from depression, anxiety, suicide, and other mental problems and we can guide them to a more positive path. 

## What we learned
We learned many things throughout the development of EUDA. We gained knowledge and experience with machine learning algorithms and neural networks, which were used to train EUDA's models. Before this project, we were virtually unaware of the possible uses of sentiment analysis, but we have now learned about sentiment analysis and how to classify the emotions conveyed in text data. We used this technique to develop the emotion detection feature of EUDA. Although we already had prior knowledge about Flask, we learned more about the Flask web framework and how to use it to build web applications. We used Flask to build the web interface for EUDA. We also discovered the uses of AJAX and JavaScript, which helped us update the front end with information from the back end in real-time. We also learned about adaptability and flexibility as we had to discard a lot of ideas that we had for this product and add other better ones. 

## What's next for EUDA
Moving forward, we plan to continue improving and expanding EUDA's capabilities by incorporating new technologies and features, such as the ability to recognize a broader range of emotions and feelings. We also want to create a 3D avatar to communicate with users. This avatar could be designed to speak out loud, using voice and facial expressions to convey emotion and empathy. This would create a more immersive and interactive experience for users, and could potentially lead to even greater engagement with EUDA. We also want to expand the product and connect it more to the community by creating online forums and support groups to help users. Overall, there are many exciting possibilities for EUDA's future development, and we look forward to continuing to improve and expand the platform in order to provide the best possible support and care for those in need.


<div align="center">


# EUDA (Emotion Understanding and Development AI)
</div>

### Cloning the repository


--> Move into the directory where we have the project files : 
```bash
cd euda

```

--> Create a virtual environment :
```bash
# Let's install virtualenv first
pip install virtualenv

# Then we create our virtual environment
virtualenv envname

```

--> Activate the virtual environment :
```bash
envname\scripts\activate

```

--> Install the requirements :
```bash
pip install -r requirements.txt

```

#

### Running the App

--> To run the App, we use :
```bash
python manage.py runserver

```

> ⚠ Then, the development server will be started at http://127.0.0.1:8000/

#


</table>



