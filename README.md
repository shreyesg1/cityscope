# CityScope - Urban Risk Intelligence Dashboard

An interactive dashboard that helps identify and visualize infrastructure and safety risks across urban neighborhoods using real-time data from NYC Open Data.

## Features

- Real-time data integration with NYC Open Data APIs
- Interactive risk map visualization
- Trend analysis for different incident types
- Risk hotspot detection and analysis
- Customizable risk weights for different factors

## Live Demo

Visit the live dashboard at: [CityScope Dashboard](https://cityscope.streamlit.app)

## Data Sources

The dashboard uses the following NYC Open Data APIs:
- 311 Service Requests
- NYPD Complaint Data
- Restaurant Inspection Results

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cityscope.git
cd cityscope
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.streamlit/secrets.toml` file with your API credentials:
```toml
nyc_open_data_token = "your_api_token"
```

4. Run the application:
```bash
streamlit run app.py
```

## Configuration

The application can be configured through the `config.json` file:
- Map settings (center coordinates, zoom level)
- API endpoints
- Risk scoring parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 