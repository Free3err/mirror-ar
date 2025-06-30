import logging
import asyncio
import tempfile
import os
import random

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import Command
from aiogram.utils.keyboard import ReplyKeyboardMarkup, KeyboardButton
from gradio_client import Client
from gradio_client.utils import handle_file

TG_TOKEN = ""

bot = Bot(TG_TOKEN)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

user_data: dict[int, dict] = {}

start_kb = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="Начать примерку")]],
    resize_keyboard=True
)

category_kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Верх")],
        [KeyboardButton(text="Низ")],
    ],
    resize_keyboard=True
)


@dp.message(Command("start"))
async def cmd_start(m: Message):
    await m.answer(
        "👋 Привет! Я помогу примерить одежду. Нажми «Начать примерку».",
        reply_markup=start_kb
    )


@dp.message(F.text == "Начать примерку")
async def init_tryon(m: Message):
    user_data[m.from_user.id] = {
        "state": "category",
        "person": None,
        "garment": None,
        "category": None
    }
    await m.answer(
        "👕 Выберите категорию одежды:",
        reply_markup=category_kb
    )


@dp.message(F.text.in_(["Верх", "Низ"]))
async def set_category(m: Message):
    uid = m.from_user.id
    if uid not in user_data or user_data[uid]["state"] != "category":
        await m.answer('Пожалуйста, сначала нажмите "Начать примерку".')
        return

    category_map = {
        "Верх": "upper_body",
        "Низ": "lower_body",
    }

    user_data[uid]["category"] = category_map[m.text]
    user_data[uid]["state"] = "person"
    await m.answer("📸 Отправьте фото человека в полный рост:", reply_markup=None)


@dp.message(F.photo)
async def receive_photo(m: Message):
    uid = m.from_user.id

    if uid not in user_data:
        await m.answer('Пожалуйста, сначала нажмите "Начать примерку".')
        return

    state = user_data[uid]["state"]

    if state == "person":
        img_bytes = await bot.download(m.photo[-1].file_id)
        user_data[uid]["person"] = img_bytes.read()
        user_data[uid]["state"] = "garment"
        await m.answer("✅ Фото человека получено. Теперь отправьте фото одежды:")

    elif state == "garment":
        img_bytes = await bot.download(m.photo[-1].file_id)
        user_data[uid]["garment"] = img_bytes.read()
        user_data[uid]["state"] = "processing"

        await m.answer("👗 Фото одежды получено. Генерирую результат...")
        await generate_result(m)

    else:
        await m.answer("⏳ Пожалуйста, дождитесь завершения текущей операции.")


@dp.message(F.text)
async def handle_other_text(m: Message):
    uid = m.from_user.id

    if uid in user_data:
        state = user_data[uid]["state"]
        if state == "category":
            await m.answer("ℹ️ Пожалуйста, выберите категорию одежды из предложенных кнопок.")
        elif state == "person":
            await m.answer("ℹ️ Пожалуйста, отправьте фото человека.")
        elif state == "garment":
            await m.answer("ℹ️ Пожалуйста, отправьте фото одежды.")
        else:
            await m.answer("⏳ Идёт обработка, пожалуйста подождите...")
    else:
        await m.answer('ℹ️ Для начала работы нажмите "Начать примерку".')


async def generate_result(m: Message):
    uid = m.from_user.id
    data = user_data[uid]

    person_path = None
    garment_path = None
    temp_files = []

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as person_file:
            person_file.write(data["person"])
            person_path = person_file.name
            temp_files.append(person_path)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as garment_file:
            garment_file.write(data["garment"])
            garment_path = garment_file.name
            temp_files.append(garment_path)

        client = Client("franciszzj/Leffa")
        client.timeout = 300

        result = client.predict(
            src_image_path=handle_file(person_path),
            ref_image_path=handle_file(garment_path),
            ref_acceleration=False,
            step=50,
            scale=2.5,
            seed=random.randint(0, 99999),
            vt_model_type="dress_code",
            vt_garment_type=data["category"],
            vt_repaint=False,
            api_name="/leffa_predict_vt"
        )

        if isinstance(result, tuple) and len(result) > 0:
            image_path = result[0]

            if isinstance(image_path, str) and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    await m.answer_photo(
                        BufferedInputFile(f.read(), filename=os.path.basename(image_path)),
                        caption="✨ Вот результат примерки!"
                    )
            else:
                logging.error(f"Некорректный путь к изображению: {image_path}")
                await m.answer("⚠️ Не удалось найти изображение.")

    except Exception as e:
        logging.exception("Ошибка при вызове API:")
        await m.answer(f"❌ Ошибка при генерации: {str(e)}")
        await m.answer("ℹ️ Возможно, модель перегружена. Попробуйте позже.")

    finally:
        for path in temp_files:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logging.warning(f"Ошибка при удалении файла {path}: {e}")

        user_data.pop(uid, None)
        await m.answer(
            "✅ Примерка завершена. Хотите попробовать ещё?",
            reply_markup=start_kb
        )


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
