import random
import requests
from io import BytesIO
import sys, os
from PIL import Image, ImageOps, ImageFont, ImageDraw
import textwrap
import json
import torch
import openai
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, \
    EulerDiscreteScheduler
from gtts import gTTS
from datetime import datetime
import gradio as gr

date_time_obj = datetime.now()
timestamp_str = date_time_obj.strftime("%Y-%m-%d_%H-%M-%S")

# ######################## gpt3 ######################


gpt3 = "text-davinci-003"  # "text-davinci-003", "text-curie-001",

try:
    openai.api_key = "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"
    completion = openai.Completion.create(
        engine="text-curie-001",
        prompt="write number 1:",
        n=1,
        max_tokens=5,
    )
    gpt3_key_succeed = True

except:
    print("openai key is not valid")
    sys.exit()

print("openai.api_key", openai.api_key)


# possible parameters: https://beta.openai.com/docs/api-reference/completions/create
def suggest_title(story, card_type, tone="funny", use_gpt3=False):
    if use_gpt3:
        model = gpt3
        max_tokens = 20
        num_texts = 3
        if story:
            prompt = f"Generate a very short {tone} {card_type} card slogan suitable for this story(in one line):"
            prompt = "story:[" + story + "] " + prompt
        else:
            prompt = f"Generate a very short {tone} {card_type} card slogan:"
        print("prompt:", prompt)
        completion = openai.Completion.create(
            engine=model,
            prompt=prompt,
            n=num_texts,
            temperature=0.5,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        results = [completion.choices[i].text.strip() for i in range(num_texts)]

        return results

    return card_type_to_title[card_type]


def improve_text(input_text, goal):
    if not input_text:
        return input_text

    prompt = f"{input_text}. {improve_text_options_dict[goal]}"
    print("prompt: ", prompt)
    model = gpt3
    max_tokens = 100
    num_texts = 1
    completion = openai.Completion.create(
        engine=model,
        prompt=prompt,
        n=num_texts,
        temperature=0.5,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    result = completion.choices[0].text.strip()

    return result


def suggest_message(story, card_type, tone="funny", use_gpt3=False):
    if use_gpt3:
        model = gpt3
        max_tokens = 50
        num_texts = 3
        if story:
            prompt = f"Generate a {tone} message for {card_type} card suitable for this story:"
            prompt = "story:[" + story + "] " + prompt
        elif story == "1":
            prompt = f"Generate a {tone} message for {card_type} card:"
        print("prompt:", prompt)
        completion = openai.Completion.create(
            engine=model,
            prompt=prompt,
            n=num_texts,
            temperature=0.5,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        results = [completion.choices[i].text.strip() for i in range(num_texts)]

        return results
    return card_type_to_message[card_type]


def generate_poem(story, card_type):
    model = gpt3
    max_tokens = 50
    num_texts = 3
    if not story or story == "1":
        # story = story_suggestions[card_type][-1]
        prompt = f"Please generate a short poem for a {card_type} card:"

    else:
        prompt = f"I want to write a poem for a {card_type} card." \
                 f" Please generate a short poem related to the following keywords/story: '{story}':"
    print("prompt:", prompt)
    completion = openai.Completion.create(
        engine=model,
        prompt=prompt,
        n=num_texts,
        temperature=0.5,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    results = [completion.choices[i].text.strip() for i in range(num_texts)]

    return results


# ######################## diffusion models ######################
diffusion_model = "runwayml/stable-diffusion-v1-5"

num_diffusion_steps = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

print("device:", device)
print("My OS:", sys.platform)

# if no GPU, use example images
if device == "cuda":

    scheduler_list = ["DPMSolverMultistep", "EulerAncestralDiscrete", "EulerDiscrete"]
    pipeline = DiffusionPipeline.from_pretrained(diffusion_model)
    pipline = pipeline.to(device)


    def generate_cards(prompt, num_images=4, scheduler="EulerDiscrete", num_steps=3):
        # generate images, return a list of (image,seed)
        # e.g [(im1, 1), (im2, 2), (im3, 3), (im4, 4)]
        images = []
        if scheduler == "EulerAncestralDiscrete":
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        elif scheduler == "DPMSolverMultistep":
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        elif scheduler == "EulerDiscrete":
            pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        # generate
        for i in range(num_images):
            seed = random.randint(0, 4294967295)
            generator = torch.Generator(device=device).manual_seed(seed)
            image = pipeline(prompt, generator=generator, num_inference_steps=num_steps).images[0]
            images.append((image, seed))

        return images


    def modify(prompt, input_image, seed, num_steps=3):
        # generate 3 images with the same seed using different schedulers
        # return 4 images, the first image is the current image
        images = [(input_image, seed)]
        for scheduler in scheduler_list:
            if scheduler == "EulerAncestralDiscrete":
                pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
            elif scheduler == "DPMSolverMultistep":
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            elif scheduler == "EulerDiscrete":
                pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

            generator = torch.Generator(device=device).manual_seed(seed)
            image = pipeline(prompt, generator=generator, num_inference_steps=num_steps).images[0]
            images.append((image, seed))

        return images


else:
    im_dir = "images"
    im1 = os.path.join(im_dir, "bookworm_1.png")
    im2 = os.path.join(im_dir, "bookworm_2.png")
    im3 = os.path.join(im_dir, "bookworm_3.png")
    im4 = os.path.join(im_dir, "bookworm_4.png")
    im_paths = [im1, im2, im3, im4]


    def generate_cards(prompt, num_images=4, scheduler="EulerDiscrete", num_steps=20):
        im_paths_ = im_paths
        random.shuffle(im_paths_)
        l = [(Image.open(card), 1) for card in im_paths_]
        return l


    def modify(prompt, input_image, seed, num_steps=20):

        l = [(Image.open(card), seed) for card in im_paths[1:]]
        return [(input_image, seed)] + l


########## text to speech #############


def text_to_speech(text, output_path="message.wav"):
    myobj = gTTS(text=text, lang='en', slow=False)
    myobj.save(output_path)
    return output_path


############################### read json files #############
json_dir = "json_files"

json_files = ['fonts.json', 'prompt_suggestions.json', 'story_suggestions.json', 'image_styles.json',
              'message_suggestions.json', 'title_suggestions.json']

json_objects = []

for json_file in json_files:
    f_path = os.path.join(json_dir, json_file)
    with open(f_path, encoding='utf-8') as f:
        json_object = json.load(f)
        json_objects.append(json_object)

font_dict, prompt_suggestions, story_suggestions, image_styles, card_type_to_message, card_type_to_title = json_objects

title_fonts = font_dict["title_fonts"]
msg_fonts = font_dict["msg_fonts"]


def add_message(input_img, recipient, sender, message, msg_font_size=25, msg_color="black"):
    """takes a chosen image with title and returns img with title and message"""
    msg_font = msg_fonts[0]
    img = input_img.copy()
    msg_req = requests.get(msg_font)
    msg_font = ImageFont.truetype(BytesIO(msg_req.content), msg_font_size)

    img_draw = ImageDraw.Draw(img)
    # left, top, right, bottom
    # border = (39, 40, 39, 156)

    w, h = input_img.size
    print("w", w)
    print("h", h)

    if h > 512 + 80:  # 200

        recipient_x = 60
        recipient_y = h - 160
    else:
        # recipient
        recipient_x = 60
        recipient_y = 512 + 30 + 5

    # apply text wrap to prevent bleed
    # msg_list = textwrap.wrap(message, 50)
    msg_new = wrap_text(message, 40)

    # msg_new = '\n'.join(msg_list)

    space = " " * 4
    recipient_msg = "Dear " + recipient + ",\n\n" + space + msg_new
    img_draw.text((recipient_x, recipient_y), recipient_msg, fill=msg_color, font=msg_font)

    # sender
    sender_x = img.size[0] - 100
    sender_y = img.size[1] - 20
    img_draw.text((sender_x, sender_y), sender, fill=msg_color, font=msg_font)

    return img


def wrap_text(text, n=4):
    # words = text.split()
    i = 0
    result = ""
    for char in text:
        # print(char)
        if (i == n):
            result += "\n"
            i = 0
        result += char
        i += 1
    return result


def add_border(img, extend_top=False, color="white"):
    # left, top, right, bottom
    border = (30, 30, 30, 200)

    if extend_top:
        border = (30, 80, 30, 200)

    # image = Image.open(img)
    image = img.copy()
    img_border = ImageOps.expand(image, border=border, fill=color)

    return img_border, extend_top


def add_title(img, title, font, pos, color="white"):
    """takes a chosen img and returns img with title"""

    req = requests.get(font)
    font_size = 1  # starting size
    title_font = ImageFont.truetype(BytesIO(req.content), font_size)

    # proportion of text wrt to image
    img_frac = 0.75
    breakpoint = img_frac * img[0].size[0]
    jumpsize = 75

    while True:

        if title_font.getsize(title)[0] < breakpoint:
            font_size += jumpsize
        else:
            jumpsize = jumpsize // 2
            font_size -= jumpsize

        title_font = ImageFont.truetype(BytesIO(req.content), font_size)
        if jumpsize <= 1:
            break

    # center text horizontally
    font_w, font_h = title_font.getsize(title)
    title_x = (img[0].size[0] - font_w) / 2

    title_y = 50

    if pos == "top":
        title_y = title_y
        if img[1] == True:
            title_y -= 33
    elif pos == "center":
        title_y = 256
    elif pos == "bottom":
        title_y = 512 - font_h
        if img[1] == True:
            title_y = 562 - font_h

    image_draw = ImageDraw.Draw(img[0])

    # draw text multiple times to create shadow
    image_draw.text((title_x - 1, title_y), title, fill="gray", font=title_font)
    image_draw.text((title_x + 1, title_y), title, fill=color, font=title_font)
    image_draw.text((title_x, title_y - 1), title, fill="gray", font=title_font)
    image_draw.text((title_x, title_y + 1), title, fill=color, font=title_font)

    return img[0]


colors = ["white", "#458B00", "#FF9912", "#5F9EA0", "#838B8B", "#EE1289"]
positions = ["top", "center", "bottom"]


def get_cards_with_diff_title_styles(input_card, title, seed):
    # apply text wrap to prevent bleed
    title = textwrap.wrap(title, 500)
    title = '\n'.join(title)

    image_with_title_list = []
    for pos in positions:
        for _ in range(3):
            # here we generate 2 styles of card with title
            # one is with title at the top,
            # the other with title at the top border
            color = random.choice(colors)
            font = random.choice(title_fonts)
            # add the border, 4 sides, top and bottom are bigger
            img_no_extend = add_border(input_card, False, "white")
            img_t1 = add_title(img_no_extend, title, font, pos, color)
            image_with_title_list.append((img_t1, seed))
            if pos == "top":
                img_no_extend = add_border(input_card, True, "white")
                img_t2 = add_title(img_no_extend, title, font, pos, color)
                image_with_title_list.append((img_t2, seed))

    return image_with_title_list


# ######################## chatbot config ######################

def enum_list(input_list):
    return [str(i) + ". " + str(item) for i, item in enumerate(input_list, 1)]


def add_bullet_points(input_list):
    return ["- " + elem for elem in input_list]


def image_path_to_image(cards):
    # cards = [(image, seed)..]
    img_list = []
    for card, seed in cards:
        # response = requests.get(card)
        # img = Image.open(BytesIO(response.content))
        if isinstance(card, str):
            card = Image.open(card)

        img_list.append(card)
    return img_list


# user options and response from bot
card_types = list(card_type_to_message.keys())

num_title_variations = 12

image_styles_options = list(image_styles)

ask_for_style_response = f"Well done! Next, choose the style of your image. Pick a number" \
                         f" or simply type in the style you want e.g. 'children book'."
message_tone = "caring"
choose_the_best_card = f" Choose the card you like the most by typing a number."

card_variations_options = ["top left", "top right", "bottom left", "bottom right"]

prompt_options = ["your idea",
                  "go back to choose a new card type"]

improve_text_options_dict = {"correct grammar": "Correct grammar mistakes in this text:",
                             "summarize": "Summarize this text:",
                             "reformulate": "Reformulate this text:",
                             "make it formal": "Make this text more formal:",
                             "make it funny": "Make this text funny:",
                             "make it informal": "Make this text more informal:",
                             "make it shorter": "Make this text shorter:",
                             "make it longer": "Make this text longer:"}

improve_text_options = ["No, I do not want to change my message."] + ["Change my message."] + list(
    improve_text_options_dict)

thank_you_message = " Enjoy! Thank you for using AICA:)"

poem_options = ["Yes, I like it.", "No, I want to use my message."]
card_finished_response = "The message is added to your card so we are finished now. "

good_bye_response_poem = "The poem is added to your card so we are finished now. "

story_example_ending = "\n\nOr type 1 to skip."

good_bye_message = "Great. We're happy you're satisfied with the card." \
                   " Now you can download it and send it as an e-card or print it out. " + thank_you_message
yes_no_options = ["yes", "no"]
title_options = ["use AICA suggestions", "your idea"]
message_options = ["use AICA suggestions", "use AICA poem generator", "your idea"]
next_steps_after_picking_best_card = ["I'm done", "I want to modify the card", "I want to add a title and a message"]

response_after_card_generated = f"Great! Thank you. We have generated 4 cards for you. " \
                                f"Let's take a look at them on the right pane."

greeting = f"Hello! I am AICA, your AI Card Artist. What kind of card would you like to generate? " \
           f"You can see the options on the right side. To choose your option, type a number (without a dot)."

# ######################## chatbot ######################

history = []
bot_actions = []
card_spec = {}


def chatbot(message, cards):
    response = ""
    print(bot_actions)
    next_steps = enum_list(next_steps_after_picking_best_card)
    print(cards)
    story_examples_ = []
    user_choices = ""
    audio_name = ""

    if "type" in card_spec:
        story_suggestions_points = add_bullet_points(story_suggestions[card_spec["type"]])
        story_examples_ = ["Tell us about good memories between you and the recipient or give us a few keywords."] + [
            "\nExamples:"] + story_suggestions_points + [story_example_ending]

    if not bot_actions or bot_actions[-1] == "final_add_audio":
        response = greeting
        bot_actions.append("greeting")
        user_choices = enum_list(card_types)

    elif bot_actions[-1] == "greeting":
        options = range(1, len(card_types) + 1)
        if message in [str(num) for num in options]:

            card_spec["type"] = card_types[int(message) - 1]
            response += f" You picked '{card_spec['type']} card'."
            response += f" Next, please tell me your name."
            bot_actions.append("asked_for_sender_name")

        else:
            response = f"Please choose a number from the right pane."
            user_choices = enum_list(card_types)

    elif bot_actions[-1] == "asked_for_sender_name":
        card_spec["sender"] = message
        response += f"Thank you. The sender's name is {message}."
        response += f" Next, please tell me the recipient's name."
        bot_actions.append("asked_for_recipient_name")

    elif bot_actions[-1] == "asked_for_recipient_name":
        card_spec["recipient"] = message
        response += f"Thank you. The recipient's name is {message}."
        response += f" Next, what image should we draw on your card?" \
                    f" We've listed a few suggestions for you." \
                    f" Please see the right pane and choose a number."
        user_choices = enum_list(prompt_suggestions[card_spec["type"]] + prompt_options)
        bot_actions.append("choose_prompt")


    elif bot_actions[-1] == "choose_prompt":
        prompt_options_all = prompt_suggestions[card_spec["type"]] + prompt_options
        options = list(range(1, len(prompt_options_all) + 1))
        if message in [str(num) for num in options]:
            if message == "4":
                response = f"Please give me a prompt. See examples on the right pane."
                user_choices = ["Examples:"] + add_bullet_points(prompt_suggestions[card_spec["type"]])
                bot_actions.append("user_gave_manual_prompt")
            elif message == "5":
                response = "Please choose a new card type by typing a number."
                bot_actions.append("greeting")
                user_choices = enum_list(card_types)
            else:
                # great, ask for style
                card_spec["user_prompt"] = prompt_options_all[int(message) - 1]
                response = ask_for_style_response
                bot_actions.append("asked_for_image_style")
                user_choices = enum_list(image_styles_options) + ["\n Or simply type your own idea."]

        else:
            response = f"Please choose a number from the right pane."
            user_choices = enum_list(prompt_options_all)

    elif bot_actions[-1] == "user_gave_manual_prompt":
        card_spec["user_prompt"] = message

        response = ask_for_style_response
        bot_actions.append("asked_for_image_style")
        user_choices = enum_list(image_styles_options) + ["\n Or simply type your own idea."]


    elif bot_actions[-1] == "asked_for_image_style":

        options = list(range(1, len(image_styles_options) + 1))
        if message in [str(num) for num in options]:  # if user pick number
            style_prompt = image_styles[image_styles_options[int(message) - 1]]
            card_spec["user_prompt"] += " " + style_prompt

        else:  # if user type his own style prompt
            card_spec["user_prompt"] += " " + message

        print("prompt to generate:", card_spec["user_prompt"])
        cards = generate_cards(card_spec["user_prompt"], num_steps=num_diffusion_steps)
        response = response_after_card_generated

        bot_actions.append("return_cards")


    elif bot_actions[-1] == "picked_best_card":
        options = range(1, len(card_variations_options) + 1)
        # next_steps = ["1. I'm finished", "2. modify the card", "3. add a title and a message"]
        if message in [str(num) for num in options]:
            card_id = int(message) - 1
            cards = [cards[card_id]]

            response = f"Good job! Now let's choose the next step."
            user_choices = next_steps  # already enum
            bot_actions.append("card_image_done")
        else:
            response = choose_the_best_card
            user_choices = enum_list(card_variations_options)

    elif bot_actions[-1] == "card_image_done":
        options = range(1, len(next_steps) + 1)
        if message in [str(num) for num in options]:
            if message == "1":
                response = good_bye_message

                bot_actions.append("final")
            elif message == "2":
                response = "You want to modify the card? Sure, then please give us a new prompt " \
                           "that contains description about things you want to change. "
                user_choices = [f"You current prompt is: '{card_spec['user_prompt']}'",
                                "------------------" * 2,
                                "Example:", "old prompt = 'A drawing of a cat'",
                                "new prompt = 'A watercolor drawing of a white cat'"]
                bot_actions.append("modify_card")
            elif message == "3":
                # response = f"What should be the title of you card? {enum_list(title_options)}"
                response = f"What should be the title of your card?"
                user_choices = enum_list(title_options)
                bot_actions.append("add_title")
        else:
            response = f"Please choose a number listed in the option box."
            user_choices = next_steps

    elif bot_actions[-1] == "modify_card":
        new_prompt = message
        card_spec["user_prompt"] = new_prompt
        old_seed = cards[-1][1]
        cards = modify(new_prompt, cards[-1][0], old_seed, num_steps=num_diffusion_steps)

        response = "Done! We've generated a few card variations for you."
        bot_actions.append("return_cards")

    elif bot_actions[-1] == "add_title":
        options = range(1, len(title_options) + 1)
        if message in [str(num) for num in options]:
            if message == "1":

                response = "Give me more information about you and the recipient of the card" \
                           " and I will suggest a few suitable card titles." \
                           " If you do not want to write a story, just type 1, and we will " \
                           "generate a few titles based on your card type:)"
                user_choices = story_examples_

                bot_actions.append("user_gave_story")
            elif message == "2":
                response = "Tell us what we should put as a title."
                bot_actions.append("user_gave_title")
        else:
            response = f"Please choose a number listed in the option box."
            user_choices = enum_list(title_options)

    elif bot_actions[-1] == "user_gave_story":
        # use story to suggest

        if message == "1":  # use AICa suggestion
            # this returns card_type_to_title[card_type]
            title_suggestions = suggest_title(message, card_spec["type"]) + ["your idea"]

        else:  # if user chose an option we provide
            story = message  # title_options[int(message)-1]
            title_suggestions = suggest_title(story, card_spec["type"], tone=message_tone, use_gpt3=True) + [
                "your idea"]
            card_spec["story"] = story
        card_spec["title_suggestions"] = title_suggestions
        response = f"Here are our suggestions for the title. Please pick a number."
        user_choices = enum_list(title_suggestions)
        bot_actions.append("user_chose_title")

    elif bot_actions[-1] == "user_chose_title":
        title_suggestions = card_spec["title_suggestions"]
        options = range(1, len(title_suggestions) + 1)
        if message in [str(num) for num in options]:
            if message == str(len(title_suggestions)):  # user picked "your idea"
                response = "Tell us what we should put as the title."
                bot_actions.append("user_gave_title")
            else:
                # put title (message) to the card, generate new card
                seed = cards[-1][1]
                card = cards[-1][0]
                title = title_suggestions[int(message) - 1]
                card_spec["title"] = title
                cards = get_cards_with_diff_title_styles(card, title, seed)

                bot_actions.append("return_card_with_title")

        else:
            response = f"Please choose a number listed in the option box."
            user_choices = enum_list(title_suggestions + ["..."])

    elif bot_actions[-1] == "user_gave_title":

        seed = cards[-1][-1]
        title = message
        card_spec["title"] = title
        card = cards[-1][0]

        cards = get_cards_with_diff_title_styles(card, title, seed)

        response += "We have put the title to your card."
        bot_actions.append("return_card_with_title")


    elif bot_actions[-1] == "picked_best_card_with_title":
        options = range(1, num_title_variations + 1)
        # next_steps = ["1. I'm finished", "2. modify the card", "3. add a title and a message"]
        if message in [str(num) for num in options]:
            card_id = int(message) - 1
            cards = [cards[card_id]]

            # response = f"Good job! Now let's choose the next step: {next_steps} "
            response = f"Do you want to put a message below your card?"

            bot_actions.append("title_done")
            user_choices = enum_list(yes_no_options)
        else:
            response = choose_the_best_card
            user_choices = list(options)


    elif bot_actions[-1] == "title_done":
        options = range(1, len(yes_no_options) + 1)
        if message in [str(num) for num in options]:
            if message == "1":  # yes
                response = f"How do you want to generate the message? Please choose a number."
                user_choices = enum_list(message_options)
                bot_actions.append("chose_message_options")
            else:  # no
                response = "Then, we are finished!" + thank_you_message
                bot_actions.append("final")
        else:
            response = f"Do you want to put a message below your card?"
            user_choices = enum_list(yes_no_options)

    elif bot_actions[-1] == "chose_message_options":
        options = range(1, len(message_options) + 1)
        if message in [str(num) for num in options]:
            if message == "1":  # user answered "use AICA suggestions"
                if "story" not in card_spec:
                    response = "Give me more information about you and the recipient of the card" \
                               " and I will suggest a personalized message for your card." \
                               " If you do not want to write a story, just type 1, and we will " \
                               "generate suggestions based on your card type:)"
                    user_choices = story_examples_
                else:
                    response = "You've already told us about your story, so I will generate suggestions based on " \
                               "this story." \
                               " Please type any key to proceed."
                bot_actions.append("user_gave_story_for_message")
            elif message == "2":  # poem
                response = "You want to generate a poem? Sure, then type in a few keywords or a story." \
                           " Or type 1 to generate a poem based only on your card type."
                user_choices = ["Example:"] + add_bullet_points(story_suggestions[card_spec["type"]])

                user_choices += [" \nOr type 1 to generate a poem based on your card type."]
                bot_actions.append("user_wants_poem")
            else:  # user pick "your idea"
                response = "Tell us what I should write under the card."
                bot_actions.append("user_gave_message")

        else:
            response = f"How do you want to generate the message? Choose a number."
            user_choices = enum_list(message_options)


    elif bot_actions[-1] == "user_wants_poem":

        response = "Here are the generated poems, please take a look and choose a number."
        poems = generate_poem(message, card_spec["type"])
        card_spec["poems"] = ["\n" + p + "\n" for p in poems]
        user_choices = enum_list(card_spec["poems"])
        bot_actions.append("user_picked_poem")

    elif bot_actions[-1] == "user_picked_poem":
        poems = card_spec["poems"]
        options = range(1, len(poems) + 1)
        if message in [str(num) for num in options]:
            poem = poems[int(message) - 1]
            card_spec["message"] = poem
            current_card = cards[-1][0]
            new_image = add_message(current_card, card_spec["recipient"], card_spec["sender"], poem)
            seed = cards[-1][-1]
            cards = [(new_image, seed)]
            response = good_bye_response_poem

            bot_actions.append("final")

        else:
            response = f"Please choose a number listed in the option box."
            user_choices = enum_list(card_spec["poems"])

    elif bot_actions[-1] == "user_gave_message":

        card_spec["message"] = message

        response = "Do you want to use our text improving service?" \
                   " See what you can do on the right pane." \
                   " Type a number to pick the next action."
        user_choices = enum_list(improve_text_options)
        bot_actions.append("improve_message")


    elif bot_actions[-1] == "improve_message":

        options = range(1, len(improve_text_options) + 1)
        if message in [str(num) for num in options]:
            if message == "1":  # No, I do not want to change my message
                bot_actions.append("add_message_to_card")

            elif message == "2":  # change message
                response = "Tell us what I should write under the card."
                bot_actions.append("user_gave_message")
            else:
                goal = improve_text_options[int(message) - 1]
                # text_improve_goal = improve_text_options_dict[goal]
                new_text = improve_text(card_spec["message"], goal)
                card_spec["improved_text"] = new_text
                response = "I've adjusted your text. You can see the result on the right pane." \
                           " Tell us if you like it and choose the next step."
                user_choices = ["Result:"] + [new_text] + ["---" * 20] + ["Next steps:"] + enum_list(poem_options)
                bot_actions.append("after_improve_text")
        else:
            response = f"Please choose a number listed in the option box."
            user_choices = enum_list(improve_text_options)

    elif bot_actions[-1] == "after_improve_text":
        options = range(1, len(poem_options) + 1)
        if message in [str(num) for num in options]:
            if message == "1":  # Yes, I like it.
                new_text = card_spec["improved_text"]
                card_spec["message"] = new_text
                current_card = cards[-1][0]
                new_image = add_message(current_card, card_spec["recipient"], card_spec["sender"],
                                        card_spec["message"])
                seed = cards[-1][-1]
                cards = [(new_image, seed)]
                response = card_finished_response
                bot_actions.append("final")
            elif message == "2":  # No, I want to use my message
                bot_actions.append("user_gave_message")
                response = "Do you want to use our text improving service?" \
                           " See what you can do on the right pane." \
                           " Type a number to pick the next action."
                user_choices = enum_list(improve_text_options)
                bot_actions.append("improve_message")

        else:
            response = f"Please choose a number listed in the option box."
            user_choices = enum_list(poem_options)


    elif bot_actions[-1] == "user_gave_story_for_message":

        if "story" not in card_spec:
            card_spec["story"] = message

        message_suggestions = suggest_message(card_spec["story"], card_spec["type"], use_gpt3=True)
        card_spec["message_suggestions"] = message_suggestions
        response = f"Here are our suggestions for the card message. Please choose a number."
        user_choices = enum_list(message_suggestions)
        bot_actions.append("user_chose_message")

    elif bot_actions[-1] == "user_chose_message":
        message_suggestions = card_spec["message_suggestions"]
        options = range(1, len(message_suggestions) + 1)
        if message in [str(num) for num in options]:
            card_message = message_suggestions[int(message) - 1]
            card_spec["message"] = card_message
            current_card = cards[-1][0]
            new_image = add_message(current_card, card_spec["recipient"], card_spec["sender"], card_message)
            seed = cards[-1][-1]
            cards = [(new_image, seed)]
            response = card_finished_response

            bot_actions.append("final")
        else:

            response = f"Please choose a number listed in the option box."
            user_choices = enum_list(message_suggestions)

    if bot_actions[-1] == "return_cards":
        response += choose_the_best_card
        bot_actions.append("picked_best_card")
        user_choices = enum_list(card_variations_options)

    if bot_actions[-1] == "return_card_with_title":
        response += choose_the_best_card
        bot_actions.append("picked_best_card_with_title")
        user_choices = enum_list(card_variations_options + ["..."])

    if bot_actions[-1] == "add_message_to_card":
        m = card_spec["message"]
        current_card = cards[-1][0]
        new_image = add_message(current_card, card_spec["recipient"], card_spec["sender"], m)
        seed = cards[-1][-1]
        cards = [(new_image, seed)]
        response += card_finished_response

        bot_actions.append("final")

    if bot_actions[-1] == "final":

        if "title" in card_spec and "message" in card_spec:
            response += " You can also click the play button below your card to read the message out loud."
            response += thank_you_message
            message_clean = card_spec["message"]
            message_clean = message_clean.replace('"', "")
            message_clean = message_clean.replace("'", "")
            text_to_say = card_spec["title"] + ". " + "Dear " + card_spec["sender"] + message_clean
            text_to_say += "from " + card_spec["recipient"]
            print("text_to_say:", text_to_say)
            audio_name = "message" + timestamp_str + ".wav"
            text_to_speech(text_to_say, output_path=audio_name)  # save speech file

        bot_actions.append("final_add_audio")

    if not response:
        response = greeting
        bot_actions.append("greeting")
        user_choices = enum_list(card_types)

    history.append((message, response))
    print(cards)

    print("...." * 20)

    no_value = ""
    card_type_str = card_spec['type'] if 'type' in card_spec else no_value
    sender_str = card_spec['sender'] if 'sender' in card_spec else no_value
    recipient_str = card_spec['recipient'] if 'recipient' in card_spec else no_value
    user_prompt_str = card_spec['user_prompt'] if 'user_prompt' in card_spec else no_value
    message_str = card_spec['message'] if 'message' in card_spec else no_value
    card_title_str = card_spec['title'] if 'title' in card_spec else no_value

    user_info = [

        f"Card type: {card_type_str}",
        f"Sender name: {sender_str}",
        f"Recipient name: {recipient_str}",
        f"Image prompt: {user_prompt_str}",
        f"\nCard title: {card_title_str}",
        f"Card message: \n{message_str}",
    ]

    user_info = "\n".join(user_info)
    user_choices = "\n".join([str(i) for i in user_choices])

    if cards:
        im = cards[-1][0]
        print(type(im))
        im.save("im.png")
    print(audio_name)
    return history, cards, user_choices, cards, user_info, audio_name


def put_audio(text):
    if text:
        return gr.update(value=text, visible=True)


################## gradio ######################

with gr.Blocks(css=".gradio-container {font-size: 20}") as demo:
    cards = gr.State([])

    with gr.Row() as row:
        with gr.Column():
            gr.Markdown(
                """
                # AICA: AI Card Artist
                Tell AICA to generate a personalized card for you!
                """)

            # https://gradio.app/docs/#chatbot
            display1 = gr.Chatbot()
            text1 = gr.Textbox(label="You", lines=1)  # gr.inputs.Textbox(label="You", lines=2)
            button1 = gr.Button(value="send", show_label=True)
            user_info = gr.Textbox(placeholder="Your card properties", visible=True, lines=1, max_lines=100)

        with gr.Column():
            gr.Markdown(
                """
                _______
                # Your card
                """)

            out_text = gr.Textbox(placeholder="Options", visible=True, lines=5, max_lines=100)

            update_cards = gr.Textbox(placeholder="temp", visible=False, lines=1, max_lines=100)
            gallery = gr.Gallery(None,
                                 label="Generated images", show_label=False, elem_id="gallery"
                                 ).style(grid=2, height="100")

            button_save = gr.Button("Save", show_label=True)
            update_audio = gr.Textbox(placeholder="temp", visible=False, lines=1, max_lines=100)

            audio = gr.Audio(visible=False)
            audio_btn = gr.Button("audio")
            f = gr.File(None, file_types=["image"], interactive=True, visible=True)

        button1.click(chatbot, scroll_to_output=True, inputs=[text1, cards], outputs=[display1, cards,
                                                                                      out_text, update_cards,
                                                                                      user_info,
                                                                                      update_audio])
        audio_btn.click(put_audio, inputs=update_audio, outputs=audio)

        button_save.click(lambda x: "im.png", inputs=[f], outputs=[f])

        update_cards.change(image_path_to_image, inputs=cards, outputs=gallery)

demo.launch(debug=True)
