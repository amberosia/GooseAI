import discord
from discord.ext import commands
from discord import app_commands
import os
import tensorflow as tf
from keras.models import load_model
import numpy as np

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=">", intents=intents)


model = load_model('gooseModel.h5')


def predict_digit(img):
    img = np.array(img)

    res = model.predict([img])[0]
    return np.argmax(res), max(res)


@bot.event
async def on_ready():
    print("We have logged in as {0.user}".format(bot))

    try:
        await bot.tree.sync()
        print("Synced!")
    except Exception as e:
        print(e)


# detect_goose slash command!
@bot.tree.command(name="detect_goose", description="Detects how Goose an image is!")
@app_commands.describe(image="Attach an image!")
async def detect_goose(interaction: discord.Interaction, image: discord.Attachment):
    if image.content_type in ('image/jpeg', 'image/png', 'image/heic'):
        try:
            await image.save(image.filename)
            image_path = image.filename

            img = tf.keras.utils.load_img(image_path, target_size=(256, 256))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            if np.argmax(score) == 0:
                percent = round(100 * np.max(score), 3)
            else:
                percent = round(100 - (100 * np.max(score)), 3)

            embed = discord.Embed(title="Goose Detector", description=f"This image is {percent}% goose!")
            embed.set_image(url=image.url)
        except:
            embed = discord.Embed(title="Goose Detector", description=":x: We ran into an unexpected error. Try again!")
    else:
        embed = discord.Embed(title="Goose Detector", description=":x: The image must be a PNG, JPG or HEIC. Try again!")

    await interaction.response.send_message(embed=embed)


try:
    bot.run(os.getenv("TOKEN"))
except discord.HTTPException as e:
    if e.status == 429:
        print("The Discord servers denied the connection for making too many requests")
        os.system("kill 1")
    else:
        raise e
