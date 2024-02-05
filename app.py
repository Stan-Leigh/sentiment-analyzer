import streamlit as st
import numpy as np
import cv2 as cv
import openai
from PIL import Image

# Page title
st.markdown("""
# Sentiment Analyzer
This app is designed to help detect sentiments of text that is given, create texts based on sentiments and compare two texts
and determine their relatability.
""")

# set the GPT-3 api key
openai.api_key = st.secrets["api_key"]

## TIPS
# https://zapier.com/blog/openai-playground/
# Use a larger temperature value to generate more diverse results.
# Use a larger max_tokens value to generate longer output.
# Use a smaller frequency_penalty value to encourage the model to use more diverse vocabulary. (0-2)
# Use a larger presence_penalty value to discourage the model from repeating itself. (0-2)

# Sentiment analyzer
text = st.text_input("Enter text that you want to analyze its sentiment")

prompt = "You are a helpful assistant that analyzes text for sentiment. You are able to call out texts that have mixed sentiments. \
          You should outputs one of these options: 'Positive', 'Negative', 'Neutral', 'Mixed' or 'Not related'."

try:
    if st.button("Get result", key="sentiment"):
        # Use GPT-3.5 to get an answer
        response = openai.ChatCompletion.create(
                                                model="gpt-3.5-turbo",
                                                messages=[{"role": "system", "content": prompt},
                                                    {"role": "user", "content": "I love pizza"},
                                                    {"role": "assistant","content": "Positive"},
                                                    {"role": "user", "content": "I love pizza but I don't like jam"},
                                                    {"role": "assistant", "content": "Mixed"},
                                                    {"role": "user", "content": text}],
                                                temperature=0,
                                                max_tokens=20,
                                                top_p=1,
                                                frequency_penalty=0,
                                                presence_penalty=0
                                                )
        # Print the result
        res = response['choices'][0]['message']['content']
        st.write(res)

except Exception as error: 
    st.warning(error)

# Text generator
st.header("Text Generator")
keywords = st.text_area("Enter as many key words that should be used in generating the text")
sentiment = st.text_input("What should be the sentiment of the text? (Positive, Negative or Neutral)")

prompt = f"Generate a tweet using the following keywords: {keywords}. The sentiment of the tweet should be {sentiment}."

try:
    if st.button("Get result", key="single"):
        # Use GPT-3.5 to get an answer
        response = openai.ChatCompletion.create(
                                                model="gpt-3.5-turbo",
                                                messages=[{"role": "user", "content": prompt}],
                                                temperature=1,
                                                max_tokens=95,
                                                top_p=1,
                                                frequency_penalty=0,
                                                presence_penalty=0
                                                )
        # Print the result
        res = response['choices'][0]['message']['content']
        st.write(res)

except Exception as error: 
    st.warning(error)


# Multi text generator
st.header("Multi-text Generator")
num_texts = st.radio("How many samples do you want to generate?", [2, 3, 4, 5], horizontal=True)
keywords = st.text_area("Enter as many key words that should be used in generating the different texts. \
                        More keywords give a more robust answer.")
sentiment = st.text_input("What should be the sentiment of the texts? (Positive, Negative or Neutral)")
max_tokens = 95 * num_texts

prompt = f"Generate {num_texts} tweets using the following keywords: {keywords}. The sentiment of the tweets should be {sentiment}."

try:
    if st.button("Get result", key="multi"):
        # Use GPT-3.5 to get an answer
        response = openai.ChatCompletion.create(
                                                model="gpt-3.5-turbo",
                                                messages=[{"role": "user", "content": prompt}],
                                                temperature=1,
                                                max_tokens=max_tokens,
                                                top_p=1,
                                                frequency_penalty=0,
                                                presence_penalty=0
                                                )
        # Print the result
        res = response['choices'][0]['message']['content']
        st.write(res)

except Exception as error: 
    st.warning(error)


# # Tweet relatability
# st.header("Text Comparison")
# st.write("Compare two blocks of text to see their relatability and sentiment.")

# text1 = st.text_area("Enter the first text")
# text2 = st.text_area("Enter the second text")

# try:
#     if st.button("Get result", key="compare"):
#         # # Sentiment analysis of the first text.
#         # prompt = f"Sentiment analysis of the following text: {text1}"

#         # response = openai.Completion.create(engine="text-davinci-003",
#         #                                         prompt=prompt,
#         #                                         temperature=0,
#         #                                         top_p=1,
#         #                                         max_tokens=20,
#         #                                         frequency_penalty=0,
#         #                                         presence_penalty=0,
#         #                                         stop=["\n"])
        
#         # # Print the result
#         # res = response["choices"][0]["text"]
#         # st.write("""#### Sentiment analysis of the first text""")
#         # st.write(res)


#         # # Sentiment analysis of the second text.
#         # prompt = f"Sentiment analysis of the following text: {text2}"

#         # response = openai.Completion.create(engine="text-davinci-003",
#         #                                         prompt=prompt,
#         #                                         temperature=0,
#         #                                         top_p=1,
#         #                                         max_tokens=20,
#         #                                         frequency_penalty=0,
#         #                                         presence_penalty=0,
#         #                                         stop=["\n"])
        
#         # # Print the result
#         # res = response["choices"][0]["text"]
#         # st.write("""#### Sentiment analysis of the second text""")
#         # st.write(res)
        
#         response = openai.Embedding.create(input=[text1, text2], model="text-embedding-ada-002")

#         embedding1 = response['data'][0]['embedding']
#         embedding2 = response['data'][1]['embedding']

#         similarity_score = np.dot(embedding1, embedding2)  # value is between -1 and 1. Higher value means more similarity.

#         st.write("""#### Text similarity""")
#         if similarity_score < -0.4:
#             st.write("The texts are complete opposites of each other.")
#         elif similarity_score < 0.4:
#             st.write("Both texts are not similar.")
#         else:
#             st.write("Both texts are similar.")

# except Exception as error: 
#     st.warning(error)


# # TEXT RELATABILITY 1
# st.header("Text Relatability")
# text1 = st.text_area("Enter the reference text")
# text2 = st.text_area("Enter the text you want to compare the reference text to")

# prompt1 = f"You are a helpful assistant that analyzes text for sentiment. You are able to call out duplicates and other forms of texts \
#             that is not genuine. You are also able to call out texts that have mixed sentiments. You should output one of these options: \
#            'Positive', 'Negative', 'Neutral', 'Mixed' or 'Not related' as the sentiment of a text with respect to the text below.\
#             \n\nText:  \"\"\"\n{text1}\n\"\"\""

# try:
#     if st.button("Get result", key="relatability"):
#         # Use GPT-3 to get an answer
#         response = openai.ChatCompletion.create(
#                                                 model="gpt-3.5-turbo",
#                                                 messages=[{"role": "system", "content": prompt1},
#                                                         {"role": "user", "content": "When is the bootcamp starting?"},
#                                                         {"role": "assistant", "content": "Neutral"},
#                                                         {"role": "user", "content": text2}],
#                                                 temperature=1,
#                                                 max_tokens=256,
#                                                 top_p=1,
#                                                 frequency_penalty=0,
#                                                 presence_penalty=0
#                                                 )

#         # Print the result
#         res = response['choices'][0]['message']['content']
#         st.write(res)

# except Exception as error: 
#     st.warning(error)


# TEXT RELATABILITY 2
st.header("Text Relatability")
text1 = st.text_area("Enter the reference text")
text2 = st.text_area("Enter the text you want to compare the reference text to")

prompt1 = f"You are a helpful assistant that analyzes text for sentiment. You will be provided a piece of text and you are to output \
            the sentiment of this text with respect to the reference text that is provided in double quotes below. \
            The reference text is \"{text1}\" \
            You can output one of these six options: \
            1. 'Positive sentiment', if the sentiment of the text with respect to the reference text is positive. \
            2. 'Negative sentiment', if the sentiment of the text with respect to the reference text is negative. \
            3. 'Not related', if the text is not related in any way to the reference text. \
            4. 'Neutral', if the sentiment of the text with respect to the reference text is neutral. \
            5. 'Mixed sentiment', if the sentiment of the text with respect to the reference text is both positive and negative. \
            6. 'Duplicated', if the text is a duplicate of itself or it contains too many repetitive words. \
        "

try:
    if st.button("Get result", key="relatability"):
        # Use GPT-3 to get an answer
        response = openai.ChatCompletion.create(
                                                model="gpt-3.5-turbo",
                                                messages=[{"role": "system", "content": prompt1},
                                                        {"role": "user", "content": "When is the bootcamp starting?"},
                                                        {"role": "assistant", "content": "Neutral"},
                                                        {"role": "user", "content": text2}],
                                                temperature=1,
                                                max_tokens=256,
                                                top_p=1,
                                                frequency_penalty=0,
                                                presence_penalty=0
                                                )

        # Print the result
        res = response['choices'][0]['message']['content']
        st.write(res)

except Exception as error: 
    st.warning(error)


# IMAGE GENERATION DALL-E 3
st.header("Image Generation")
text = st.text_input("Enter the description of the image you want to create", "a white siamese cat")
image_size = st.radio("What size should the generated image have?", ['1024x1024', '1024x1792', '1792x1024'], horizontal=True)

try:
    response = openai.Image.create(
    model="dall-e-3",
    prompt=text,
    size=image_size,
    quality="standard",
    n=1,
    )

except Exception as error: 
    st.warning(error)

image_url = response.data[0].url
st.image(image_url, caption=text)

# # IMAGE GENERATION DALL-E 2
# st.header("Image Generation")
# text = st.text_input("Enter the description of the image you want to create", "a white siamese cat")
# image_size = st.radio("What size should the generated image have?", ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024'], horizontal=True)

# try:
#     response = openai.Image.create(
#     model="dall-e-2",
#     prompt=text,
#     size='512x512',
#     quality="standard",
#     n=1,
#     )

# except Exception as error: 
#     st.warning(error)

# image_url = response.data[0].url
# st.image(image_url, caption=text)


# # IMAGE EDITING DALL-E 2
# st.header("Image Editing")
# text = st.text_input("Enter the description of the image, including parts you want edited inside.", "An indoor lounge area with a brown dog swimming in the pool")
# uploaded_file = st.file_uploader("Upload your image in .png format ", type=["png"], key='file')
# uploaded_mask = st.file_uploader("Upload your image with mask in .png format ", type=["png"], key='mask')
# image_size = st.radio("What size should the generated image have?", ['256x256', '512x512', '1024x1024'], horizontal=True)

# # get resizing tuple
# resize_cv = 0
# if image_size == '256x256':
#     resize_cv = (256, 256)
# elif image_size == '512x512':
#     resize_cv = (512, 512)
# else:
#     resize_cv = (1024, 1024)

# # Process the files uploaded
# # image_cv = cv.imdecode(np.asarray(bytearray(uploaded_file.read()), np.uint8), 1)
# # image_mask_cv = cv.imdecode(np.asarray(bytearray(uploaded_mask.read()), np.uint8), 1)


# if uploaded_file is not None and uploaded_mask is not None:
#     # Process the file uploaded
#     image = Image.open(uploaded_file)
#     img_array = np.array(image)
#     cv.imwrite('file.png', cv.cvtColor(img_array, cv.COLOR_RGB2BGR))

#     # Resize the image
#     image_cv = cv.imread('file.png')
#     image_cv = cv.resize(image_cv, resize_cv)

#     # Process the mask uploaded
#     mask = Image.open(uploaded_mask)
#     mask_array = np.array(mask)
#     cv.imwrite('mask.png', cv.cvtColor(mask_array, cv.COLOR_RGB2BGR))

#     # Resize the mak
#     image_mask_cv = cv.imread('mask.png')
#     image_mask_cv = cv.resize(image_mask_cv, resize_cv)

#     try:
#         response = openai.Image.create_edit(
#             model="dall-e-2",
#             image=image_cv,
#             mask=image_mask_cv,
#             prompt=text,
#             n=1,
#             size=image_size
#         )

#     except Exception as error: 
#         st.warning(error)

#     # Get output
#     image_url = response.data[0].url
#     st.image(image_url)
# else:
#     st.write("Upload a file and its masked file")

# The uploaded image and mask must both be square PNG images less than 4MB in size, 
# and also must have the same dimensions as each other. 
# The non-transparent areas of the mask are not used when generating the output, 
# so they don’t necessarily need to match the original image like the example above.