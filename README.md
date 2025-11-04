# ReviewSense
## What is ReviewSense?



ReviewSense is an AI-powered app that analyzes customer reviews to help businesses understand what people think about their products. It works with both English and Arabic reviews and can detect whether feedback is positive, negative, or neutral. The app also pulls out the main topics that customers are talking about.



---



## Main Features



- Analyzes customer sentiment (positive, negative, or neutral)

- Extracts key topics from reviews

- Supports both English and Arabic text

- Detects Arabic dialects

- User login and registration

- Saves your analysis history

- Clean, easy-to-use interface

- Fast processing with real-time results



---



## How It Works



The project has two main parts:



*Frontend (What you see)*

- Built with React and TypeScript

- Modern dashboard design

- Connects to either Google's Gemini AI or our own backend server



*Backend (The processing part)*

- Flask server with PostgreSQL database

- Uses Google Gemini API and Hugging Face models

- Handles user accounts and saves analysis sessions



---



## What You Need



- Node.js (version 16 or higher)

- Python 3.8+

- PostgreSQL database

- Google Gemini API key



---



## Setting It Up



*Frontend:*

bash

npm install

npm run dev



Then go to http://localhost:5173



*Backend:*

bash

cd flask_backend

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt





Create a .env file with:



GEMINI_API_KEY=your_api_key_here

DATABASE_URL=PostgreSQL://postgres:postgres@localhost:5432/reviewsense

SECRET_KEY=your-secret-key





Then run:

bash

flask db upgrade

python app.py





---



## How to Use It



1. Start both the frontend and backend

2. Open http://localhost:5173 in your browser

3. Create an account or log in

4. Upload or paste your reviews

5. Click "Analyze Reviews"

6. View the results showing sentiment, topics, and dialect (for Arabic)

7. Check your past analyses from the dashboard



---



## Project Files





reviewsense-main/

├── components/          # React components

├── services/           # API connections

├── flask_backend/      # Backend server

├── sentiment-analysis-service/

├── sentiment-analyzer/

└── configuration files





---



## Deployment



*For Production:*



Frontend:

bash

npm run build



Then upload the dist folder to your hosting service.



Backend:

bash

pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 app:app





---



## Security Features



- API keys stored securely

- Password encryption

- Protected API endpoints

- SQL injection prevention

- Input validation



---



## Common Problems



If something doesn't work:

- Make sure both servers are running

- Check your API key is correct

- Verify database connection

- Try reinstalling: rm -rf node_modules && npm install



---



## Credits



This project was developed for my graduation from the IT program at Qassim University. Built using Flask, React, Google Gemini AI, and Hugging Face Transformers.