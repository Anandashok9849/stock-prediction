# 📈 Stock Price Prediction Web App

A sophisticated Flask web application that predicts stock prices using Long Short-Term Memory (LSTM) neural networks. The app provides real-time stock analysis, price predictions, interactive charts, and news integration.

## 🌟 Features

- **LSTM Neural Network**: Advanced deep learning model for accurate stock price predictions
- **Real-time Data**: Fetches live stock data using Yahoo Finance API
- **Interactive Charts**: Beautiful visualizations using Plotly for market analysis
- **News Integration**: Latest news articles related to stocks using NewsAPI
- **Web Interface**: User-friendly Flask web application
- **Performance Metrics**: MSE and RMSE calculations for model evaluation
- **Next Day Prediction**: Predicts stock prices for the next trading day

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Flask** - Web framework
- **TensorFlow/Keras** - Deep learning framework
- **Scikit-learn** - Machine learning utilities
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Plotly** - Data visualization
- **Yahoo Finance API** - Stock data fetching
- **NewsAPI** - News integration
- **Joblib** - Model serialization

## 📋 Prerequisites

- Python 3.8 or higher
- Git
- Internet connection for data fetching

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Anandashok9849/stock-prediction.git
   cd stock-prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up NewsAPI key:**
   - Get your API key from [NewsAPI](https://newsapi.org/)
   - Replace the API key in `app.py`:
   ```python
   API_KEY = "YOUR_NEWSAPI_KEY_HERE"
   ```

## 📊 Training the Model

To train your own model:

1. Run the training script:
   ```bash
   python train_model.py
   ```

2. This will:
   - Download AAPL stock data from 2010
   - Train an LSTM model
   - Save the model as `model.h5`
   - Save the scaler as `scaler.pkl`

## 🌐 Usage

1. **Run the Flask application:**
   ```bash
   python app.py
   ```

2. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

3. **Enter a stock symbol** (e.g., AAPL, GOOGL, MSFT) and click "Predict"

## 📁 Project Structure

```
stock-prediction/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── model.h5              # Trained LSTM model
├── scaler.pkl            # Data scaler
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku deployment config
├── templates/
│   └── index.html        # Web interface template
└── static/               # Static files (charts, images)
    └── result.png        # Generated prediction charts
```

## 🎯 How It Works

1. **Data Collection**: Fetches historical stock data using yfinance
2. **Data Preprocessing**: Scales data using MinMaxScaler
3. **Model Prediction**: Uses trained LSTM model to predict future prices
4. **Visualization**: Creates interactive charts and prediction plots
5. **News Integration**: Fetches relevant news articles
6. **Web Display**: Presents results in a user-friendly interface

## 📈 Model Architecture

- **Input Layer**: 100 time steps of stock prices
- **LSTM Layer 1**: 50 units with return sequences
- **Dropout Layer 1**: 20% dropout for regularization
- **LSTM Layer 2**: 50 units
- **Dropout Layer 2**: 20% dropout
- **Dense Layer**: 1 unit for price prediction
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error

## 🚀 Deployment

### Heroku Deployment

1. **Install Heroku CLI**
2. **Login to Heroku:**
   ```bash
   heroku login
   ```

3. **Create Heroku app:**
   ```bash
   heroku create your-app-name
   ```

4. **Deploy:**
   ```bash
   git push heroku master
   ```

### Render Deployment

1. **Connect your GitHub repository to Render:**
   - Go to [Render.com](https://render.com) and sign up/login
   - Click "New +" and select "Web Service"
   - Connect your GitHub account and select the `stock-prediction` repository

2. **Configure the service:**
   - **Name**: `stock-prediction` (or your preferred name)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

3. **Environment Variables:**
   - Add your NewsAPI key: `NEWS_API_KEY=your_actual_api_key_here`

4. **Deploy:**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app

### Local Deployment

The app includes a `Procfile` for easy deployment to platforms like Heroku, Render, or Railway.

## 📊 Performance Metrics

- **Mean Squared Error (MSE)**: Measures average squared difference between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Square root of MSE, in same units as the target variable

## 🔧 Configuration

### NewsAPI Setup
Replace the placeholder API key in `app.py` with your actual NewsAPI key:
```python
API_KEY = "your_actual_api_key_here"
```

### Model Parameters
Modify `train_model.py` to adjust:
- Number of epochs
- Batch size
- Stock symbol for training
- Date range for data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for stock data
- NewsAPI for news integration
- TensorFlow/Keras for deep learning framework
- Flask community for web framework

## 📞 Support

If you have any questions or issues, please open an issue on GitHub or contact the repository owner.

---

**Note**: This is an educational project for demonstration purposes. Stock price prediction is inherently uncertain and should not be used for actual financial decision-making.</content>
<parameter name="filePath">c:\Users\Winsows\Desktop\vs code\Stock Price Prediction Updated\README.md