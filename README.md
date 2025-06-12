This voice translation tool supports translation, transcription, and audio translation for all languages. It's suitable for the AI era that emphasizes speed. It might not translate the nuances of tone as well as the original, perhaps because current AI still needs a bit more development. Looking at this tool, it seems more suitable for game voice modders, 555+.

🚴‍♂️Currently, it has the following functions:

1️⃣Voice analysis for gender separation

            ✅While transcribing, distinguishes between male and female voices (60-70% accuracy)

2️⃣Transcription with 2 main types, selectable for languages worldwide

            ✅Transcribe with Whisper, with 5 sub-levels of transcription detail (free)
            
            ✅Transcribe with Google Cloud, requires .json key from Google (paid)
            
            ✅Transcribe with Gemini, requires writing key in .ini file (supports Multi key) (both free and paid, depending on the Key)
            
            ✅Supports other APIs, currently no API supported for this transcription part (reserved)

3️⃣ Translation

            ✅Can directly translate using Google (selectable for languages worldwide), or import translated TXT from elsewhere (free)

4️⃣Voice synthesis, can choose the output audio format

               ✅Options include gTTS (free) and Google Cloud (paid)

               ✅Can add channels as needed (1 channel = 1 voice)

               ✅Currently only supports Google Cloud (requires .json key)
               
📌 📌 📌 📌 📌 📌 📌 Required files :📌 📌 📌 📌

               ✅VC_redist.x64.exe ( https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 )
               
               ✅VoicesAll.txt (For the list of each sound, you can remove the lines for countries that are not needed.)

               ✅settings.ini (For Gemini Key)
               
‼️⁉️‼️⁉️Warning: ‼️⁉️‼️⁉️‼️⁉️‼️⁉️

            ‼️⁉️‼️⁉️Warning: Right-click the .exe file and select "Run as administrator" only.‼️⁉️‼️⁉️‼️⁉️‼️⁉️
            And because the .exe file is large, it may take a moment for the program window to appear after you run it

================================
# Installation Guide
================================

            ### Step 1: Install Visual C++ Redistributable

            - Download and install the latest supported Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, 2019, and 2022.
            - Get the x64 version from this link: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

            ### Step 2: Restart Your Computer

            -  After the installation is complete, please restart your PC.

            ### Step 3: Download a Whisper Model

            - Choose and download one of the following Whisper models. The models are listed by size and capability. Larger models offer higher accuracy but require more computational resources.
            - The model files should be placed in the appropriate directory for the application to use them.

            **Official Model Download Links:**

            -   **tiny.en**: https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt
            -   **tiny**: https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt
            -   **base.en**: https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt
            -   **base**: https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt
            -   **small.en**: https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt
            -   **small**: https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt
            -   **medium.en**: https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt
                        -   **medium**: https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt
            -   **large-v1**: https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt
            -   **large-v2**: https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt
            -   **large-v3**: https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt
            -   **large**: https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt (Same as large-v3)


            *Note: The original source for these model links can be found in the Whisper repository:*
            * https://github.com/openai/whisper/blob/main/whisper/__init__.py

📌 Channel will be specified in the .txt after transcription as Gender|0|0|, where the third |0| is for specifying the Channel to be used.

📌 📌 In the voice creation section, the free Google voice has only one female voice. No issues found with the length of the generated audio (while Google Cloud seems to have a character limit for audio synthesis, likely around 100 characters).

This tool is complete and ready to use if AI is not used. However, if you want to use AI, you will need additional keys and files, such as:

            1. Google Cloud .json key (you need to sign up for Google Cloud to download the .json key file).
            2. Gemini requires a settings.ini file. Actually, you can use any name; just create a new .txt file and change its extension to .ini, which allows you to enter both free and paid Gemini API Keys.
            3. Voice files: You can copy the names of each voice from https://cloud.google.com/text-to-speech/docs/list-voices-and-types and paste them into a .txt file.

For example:

             Afrikaans (South Africa)	Standard	af-ZA	af-ZA-Standard-A	FEMALE
             Arabic	Premium	ar-XA	ar-XA-Chirp3-HD-Aoede	FEMALE
             Arabic	Premium	ar-XA	ar-XA-Chirp3-HD-Charon	MALE
             Arabic	Premium	ar-XA	ar-XA-Chirp3-HD-Fenrir	MALE

 You can find JSON KEY by typing this sentence into YouTube.

                       How to Get JSON Key from Google Cloud Console

How to use and various functions : (https://youtu.be/SJyh7CW54WY)


🥺I'm really having financial difficulties lately, not enough to cover rent and electricity bills.
If you find this tool somewhat useful, you can do so at this 🥺

USDT QR Code : 

            THu1RcJQcxKqo2ePdChHWKp6S3dxXbgBA3

or 

 https://imgbiz.com/image/494687746-1034091198188409-2650559689811165240-n.WJX2M 

สำหรับคนไทยอยากช่วยเหลือ สามารถโดเนทได้ที่ SCB 418-080-4938 หรือ Facebook.com/maimaomaiplae](https://www.facebook.com/maimaomaiplae)
