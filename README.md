# Electricity Consumption Patterns in Israeli Households  

This repository contains the code and analysis for ongoing research into electricity consumption patterns in Israeli households.
The study aims to uncover unique usage behaviors, device-specific energy profiles, and temporal consumption trends to better understand residential energy dynamics.  

## Project Objectives  
- Identify distinct electricity usage patterns in Israeli households.  
- Analyze device-specific energy consumption, focusing on builers (electrice water heaters), air conditioners, and other high-power devices.  
- Develop efficient segmentation and classification methods for power events.  
- Create visualizations to showcase consumption trends and anomalies.  

## Features  
- **Data Segmentation:** Techniques to split raw data into meaningful segments based on power thresholds and time windows.  
- **Event Classification:** Algorithms to identify and categorize power events, such as heater activations.  
- **Visualization Tools:** Graphs and charts to illustrate patterns, including phase-specific consumption and time-of-day trends.  

## Repository Structure  
- **/code**: Contains Python scripts for data analysis, event detection, and visualization.  
- **/data**: Example datasets (sanitized for privacy).  
- **/plots**: Saved visualizations generated during analysis.  
- **/docs**: Documentation on methods, assumptions, and findings.  

## Getting Started  
### Prerequisites  
- Python 3.8 or higher.  
- Required packages: pandas, numpy, matplotlib (detailed in `requirements.txt`).  

### Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/username/repo-name.git  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

### Usage  
1. Place raw data files in the `/data` directory.  
2. Run `analyze.py` to segment and classify events:  
   ```bash  
   python analyze.py  
   ```  
3. View saved visualizations in the `/plots` directory.
