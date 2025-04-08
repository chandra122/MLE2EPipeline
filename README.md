# CO2 Emission Prediction System

This project is a full-stack application that predicts CO2 emissions based on various input parameters. It consists of a React frontend and a Flask backend with machine learning models.

## Project Structure

```
MLE2EPipeline/
├── frontend/           # React frontend application
│   ├── src/           # Source code
│   ├── public/        # Static files
│   └── package.json   # Frontend dependencies
├── backend/           # Flask backend application
│   ├── app.py         # Main Flask application
│   ├── requirements.txt # Python dependencies
│   └── *.joblib       # Trained machine learning models
```

## Features

- Interactive web interface for CO2 emission prediction
- Multiple machine learning models (Random Forest, Linear Regression, Decision Tree)
- Data visualization using Chart.js and Recharts
- RESTful API endpoints for predictions
- Modern UI built with Material-UI

## Technologies Used

### Frontend
- React 19
- Material-UI (MUI) for UI components
- Chart.js and Recharts for data visualization
- Axios for API communication

### Backend
- Flask 3.1.0
- Scikit-learn for machine learning models
- Pandas for data manipulation
- Flask-CORS for handling cross-origin requests
- Gunicorn for production deployment

## Getting Started

### Prerequisites
- Node.js (for frontend)
- Python 3.x (for backend)
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd MLE2EPipeline
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
```

3. Install backend dependencies:
```bash
cd ../backend
pip install -r requirements.txt
```

### Running the Application

1. Start the backend server:
```bash
cd backend
python app.py
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

The application will be available at `http://localhost:3000`

## Machine Learning Models

The backend includes three trained models:
- Random Forest Model
- Linear Regression Model
- Decision Tree Model

These models are trained on the CO2_emission.csv dataset and are used to make predictions based on user input.

## API Endpoints

The backend provides the following endpoints:
- `/predict` - Main prediction endpoint
- Additional endpoints for model-specific predictions

## Deployment

The application includes Procfile configurations for both frontend and backend, making it ready for deployment on platforms like Heroku.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 