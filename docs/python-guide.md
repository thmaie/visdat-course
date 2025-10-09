# Python Programming Guide

## Overview

Python is a high-level, interpreted programming language that excels in data processing, analysis, and visualization. Its clear syntax, extensive standard library, and rich ecosystem make it an ideal choice for rapid prototyping and data science applications.

## Why Python for Data Processing?

### Advantages

- **Productivity:** Clean, readable syntax accelerates development
- **Rich Ecosystem:** Extensive libraries for data science (NumPy, Pandas, Matplotlib)
- **Interactive Development:** Jupyter notebooks for exploratory analysis
- **Cross-Platform:** Runs on Windows, Linux, macOS
- **Community:** Large, active community with extensive documentation

### Use Cases in This Course

- **Data Analysis:** Processing and cleaning datasets
- **Visualization:** Creating charts, plots, and interactive dashboards
- **Prototyping:** Rapid development of data processing algorithms
- **Automation:** Scripting repetitive data tasks
- **Integration:** Connecting different data sources and formats

## Python Basics

### Language Characteristics

Python is an interpreted language with several key features:

- **Dynamic Typing:** Variable types are determined at runtime
- **Indentation-Based:** Code blocks are defined by indentation, not braces
- **Interactive:** Can be run interactively or as scripts
- **Object-Oriented:** Everything is an object

### Basic Syntax

```python
# Comments start with #

# Variables and basic types
integer_var = 42
float_var = 3.14159
string_var = "Hello, Python!"
boolean_var = True

# Print output
print(f"Integer: {integer_var}")
print(f"Float: {float_var}")
print(f"String: {string_var}")
print(f"Boolean: {boolean_var}")

# Multiple assignment
x, y, z = 1, 2, 3
print(f"x={x}, y={y}, z={z}")
```

### Data Types and Structures

#### Built-in Data Types

```python
# Numbers
integer = 42
floating_point = 3.14159
complex_number = 3 + 4j

print(f"Integer: {integer}, type: {type(integer)}")
print(f"Float: {floating_point}, type: {type(floating_point)}")
print(f"Complex: {complex_number}, type: {type(complex_number)}")

# Strings
single_quotes = 'Hello'
double_quotes = "World"
multi_line = """This is a
multi-line string"""

# String operations
name = "Python"
print(f"Length: {len(name)}")
print(f"Uppercase: {name.upper()}")
print(f"Replace: {name.replace('Python', 'Programming')}")

# String formatting
temperature = 22.5
print(f"Temperature: {temperature:.1f}°C")
print("Temperature: {:.1f}°C".format(temperature))
```

#### Collections

```python
# Lists - ordered, mutable collections
numbers = [1, 2, 3, 4, 5]
mixed_list = [1, "hello", 3.14, True]

# List operations
numbers.append(6)                    # Add element
numbers.insert(0, 0)                # Insert at position
numbers.remove(3)                   # Remove first occurrence
popped = numbers.pop()              # Remove and return last element

print(f"Numbers: {numbers}")
print(f"First three: {numbers[:3]}")
print(f"Last three: {numbers[-3:]}")

# List comprehensions
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

print(f"Squares: {squares}")
print(f"Even squares: {even_squares}")
```

```python
# Tuples - ordered, immutable collections
coordinates = (3.14, 2.71)
point = ("New York", 40.7128, -74.0060)

# Tuple unpacking
city, latitude, longitude = point
print(f"City: {city}, Lat: {latitude}, Lon: {longitude}")

# Named tuples for structured data
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p1 = Point(1, 2)
print(f"Point: x={p1.x}, y={p1.y}")
```

```python
# Dictionaries - key-value pairs
sensor_data = {
    "temperature": 22.5,
    "humidity": 65.0,
    "pressure": 1013.25
}

# Dictionary operations
sensor_data["timestamp"] = "2023-10-01T10:30:00"
sensor_data.update({"location": "Room A", "status": "active"})

print(f"Temperature: {sensor_data['temperature']}°C")
print(f"Keys: {list(sensor_data.keys())}")
print(f"Values: {list(sensor_data.values())}")

# Dictionary comprehensions
squared_dict = {x: x**2 for x in range(5)}
print(f"Squared dict: {squared_dict}")
```

```python
# Sets - unordered collections of unique elements
unique_numbers = {1, 2, 3, 4, 5}
another_set = {4, 5, 6, 7, 8}

# Set operations
union = unique_numbers | another_set
intersection = unique_numbers & another_set
difference = unique_numbers - another_set

print(f"Union: {union}")
print(f"Intersection: {intersection}")
print(f"Difference: {difference}")
```

## Control Structures

### Conditional Statements

```python
def categorize_temperature(temp):
    """Categorize temperature reading."""
    if temp < 0:
        return "Freezing"
    elif temp < 10:
        return "Cold"
    elif temp < 25:
        return "Mild"
    elif temp < 35:
        return "Warm"
    else:
        return "Hot"

# Usage
temperature = 22.5
category = categorize_temperature(temperature)
print(f"{temperature}°C is {category}")

# Ternary operator
status = "High" if temperature > 30 else "Normal"
print(f"Status: {status}")

# Multiple conditions
age = 25
has_license = True

if age >= 18 and has_license:
    print("Can drive")
elif age >= 16 and has_license:
    print("Can drive with supervision")
else:
    print("Cannot drive")
```

### Loops

```python
# For loops
measurements = [12.5, 15.2, 11.8, 16.1, 13.9]

# Iterate over values
print("Temperature readings:")
for temp in measurements:
    print(f"  {temp}°C")

# Iterate with index
print("\nIndexed readings:")
for i, temp in enumerate(measurements):
    print(f"  Reading {i+1}: {temp}°C")

# Iterate over range
print("\nRange iteration:")
for i in range(5):
    print(f"Index: {i}")

# Range with start, stop, step
for i in range(0, 10, 2):
    print(f"Even number: {i}")
```

```python
# While loops
import random

# Simulate sensor readings until target reached
target_temp = 25.0
current_temp = 20.0
iterations = 0

while abs(current_temp - target_temp) > 0.1 and iterations < 100:
    # Simulate temperature change
    change = random.uniform(-0.5, 1.0)
    current_temp += change
    iterations += 1
    
    print(f"Iteration {iterations}: {current_temp:.2f}°C")

print(f"Reached target after {iterations} iterations")
```

```python
# Loop control statements
data = [1, 2, 0, 4, -1, 6, 7, 8, 0, 10]

print("Processing data with continue and break:")
for i, value in enumerate(data):
    if value == 0:
        print(f"  Skipping zero at index {i}")
        continue
    
    if value < 0:
        print(f"  Found negative value {value} at index {i}, stopping")
        break
    
    print(f"  Processing value {value}")
```

## Virtual Environments

### Why Virtual Environments?

Virtual environments are isolated Python environments that allow you to:
- **Avoid conflicts** between package versions for different projects
- **Keep your system Python clean** and stable
- **Share exact dependencies** with team members
- **Test different Python versions** for the same project

> **📽️ Quick Reference:** See Lecture 1 Slides for basic commands.

### Creating and Using Virtual Environments

#### Step 1: Create a Virtual Environment

```bash
# Navigate to your project directory
cd my-project

# Create virtual environment
python -m venv .venv

# Alternative: specify Python version (if multiple installed)
python3.9 -m venv .venv
```

#### Step 2: Activate the Virtual Environment

**Windows:**
```cmd
# Command Prompt
.venv\Scripts\activate

# PowerShell
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**You'll see the environment name in your prompt:**
```bash
(.venv) user@computer:~/my-project$
```

#### Step 3: Install Packages

```bash
# With virtual environment activated
pip install numpy pandas matplotlib

# Install specific versions
pip install numpy==1.21.0

# Install from requirements file
pip install -r requirements.txt
```

#### Step 4: Deactivate When Done

```bash
deactivate
```

### VS Code Integration

VS Code automatically detects virtual environments:

1. **Open project folder** in VS Code
2. **Python interpreter selection:**
   - `Ctrl+Shift+P` → "Python: Select Interpreter"
   - Choose `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (macOS/Linux)
3. **Terminal integration:** New terminals automatically activate the environment

### Managing Dependencies

#### Create requirements.txt

```bash
# Generate requirements file
pip freeze > requirements.txt
```

**Example requirements.txt:**
```
numpy==1.21.6
pandas==1.4.2
matplotlib==3.5.2
plotly==5.8.0
pyvista==0.35.2
```

#### Install from requirements.txt

```bash
# In new environment
pip install -r requirements.txt
```

### Common Workflows

#### New Project Setup

```bash
# 1. Create project directory
mkdir my-data-project
cd my-data-project

# 2. Create virtual environment
python -m venv .venv

# 3. Activate environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 4. Install dependencies
pip install numpy pandas matplotlib jupyter

# 5. Create requirements file
pip freeze > requirements.txt

# 6. Start coding!
```

#### Sharing Project with Team

```bash
# Include in your Git repository:
# ✅ requirements.txt
# ❌ .venv/ (add to .gitignore)

# Team member setup:
git clone your-repo
cd your-repo
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Best Practices

1. **One environment per project**
2. **Always activate before installing packages**
3. **Use descriptive names** for complex setups:
   ```bash
   python -m venv venv-data-analysis
   ```
4. **Keep requirements.txt updated**
5. **Add .venv/ to .gitignore**

### Troubleshooting

#### Environment Not Activating
```bash
# Windows: Try different activation scripts
.venv\Scripts\activate.bat
.venv\Scripts\activate.ps1
.venv\Scripts\Activate.ps1

# Check PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Wrong Python Version
```bash
# Check current Python
python --version

# Create with specific Python
python3.9 -m venv .venv
```

#### Package Installation Fails
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Then install packages
pip install package-name
```

### Alternative: Conda Environments

For data science projects, Conda is also popular:

```bash
# Create conda environment
conda create -n myenv python=3.9

# Activate
conda activate myenv

# Install packages
conda install numpy pandas matplotlib

# Deactivate
conda deactivate
```

## Functions

### Basic Functions

```python
def calculate_statistics(data):
    """
    Calculate basic statistics for a dataset.
    
    Args:
        data (list): List of numerical values
        
    Returns:
        dict: Dictionary containing statistical measures
        
    Raises:
        ValueError: If data is empty
    """
    if not data:
        raise ValueError("Cannot calculate statistics for empty dataset")
    
    n = len(data)
    mean = sum(data) / n
    
    # Calculate variance and standard deviation
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = variance ** 0.5
    
    # Find min and max
    minimum = min(data)
    maximum = max(data)
    
    return {
        'count': n,
        'mean': mean,
        'std_dev': std_dev,
        'min': minimum,
        'max': maximum,
        'range': maximum - minimum
    }

# Usage
measurements = [12.5, 15.2, 11.8, 16.1, 13.9, 14.7, 12.3]
stats = calculate_statistics(measurements)

print("Dataset Statistics:")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")
```

### Function Parameters

```python
def process_data(data, operation="mean", precision=2, verbose=False):
    """
    Process data with various options.
    
    Args:
        data (list): Input data
        operation (str): Operation to perform ('mean', 'sum', 'max', 'min')
        precision (int): Decimal places for output
        verbose (bool): Whether to print detailed output
    """
    if verbose:
        print(f"Processing {len(data)} data points with operation '{operation}'")
    
    if operation == "mean":
        result = sum(data) / len(data)
    elif operation == "sum":
        result = sum(data)
    elif operation == "max":
        result = max(data)
    elif operation == "min":
        result = min(data)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return round(result, precision)

# Different ways to call the function
data = [1.234, 2.567, 3.891, 4.123]

# Positional arguments
result1 = process_data(data, "mean", 3, True)

# Keyword arguments
result2 = process_data(data, operation="max", verbose=True)

# Mixed arguments
result3 = process_data(data, "sum", verbose=True)

print(f"Results: {result1}, {result2}, {result3}")
```

### Lambda Functions

```python
# Lambda functions for simple operations
square = lambda x: x ** 2
add = lambda x, y: x + y

print(f"Square of 5: {square(5)}")
print(f"Sum of 3 and 4: {add(3, 4)}")

# Using lambdas with built-in functions
numbers = [1, 2, 3, 4, 5]

# Map - apply function to all elements
squared = list(map(lambda x: x**2, numbers))
print(f"Squared: {squared}")

# Filter - select elements based on condition
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Even numbers: {evens}")

# Sorting with custom key
data = [("Alice", 85), ("Bob", 90), ("Charlie", 78)]
sorted_by_score = sorted(data, key=lambda x: x[1])
print(f"Sorted by score: {sorted_by_score}")
```

## File Handling

### Reading Files

```python
def read_temperature_data(filename):
    """Read temperature data from a text file."""
    temperatures = []
    
    try:
        with open(filename, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    try:
                        temp = float(line)
                        temperatures.append(temp)
                    except ValueError:
                        print(f"Warning: Invalid data on line {line_num}: {line}")
        
        print(f"Successfully read {len(temperatures)} temperature readings")
        return temperatures
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return []
    except IOError:
        print(f"Error: Could not read file '{filename}'")
        return []

# Create sample data file
sample_data = """# Temperature readings (Celsius)
22.5
23.1
21.8
24.2
22.9
invalid_data
25.1
23.7
"""

with open('temperature_data.txt', 'w') as f:
    f.write(sample_data)

# Read the data
temperatures = read_temperature_data('temperature_data.txt')
if temperatures:
    stats = calculate_statistics(temperatures)
    print(f"Average temperature: {stats['mean']:.1f}°C")
```

### Writing Files

```python
def save_analysis_report(data, filename, include_raw_data=True):
    """Save data analysis report to a file."""
    stats = calculate_statistics(data)
    
    with open(filename, 'w') as file:
        file.write("Data Analysis Report\n")
        file.write("=" * 20 + "\n\n")
        
        file.write(f"Dataset Summary:\n")
        file.write(f"  Number of readings: {stats['count']}\n")
        file.write(f"  Mean: {stats['mean']:.2f}\n")
        file.write(f"  Standard deviation: {stats['std_dev']:.2f}\n")
        file.write(f"  Minimum: {stats['min']:.2f}\n")
        file.write(f"  Maximum: {stats['max']:.2f}\n")
        file.write(f"  Range: {stats['range']:.2f}\n\n")
        
        if include_raw_data:
            file.write("Raw Data:\n")
            for i, value in enumerate(data, 1):
                file.write(f"  {i:2d}: {value:6.2f}\n")
    
    print(f"Report saved to '{filename}'")

# Generate and save report
save_analysis_report(temperatures, 'analysis_report.txt')
```

### CSV File Handling

```python
import csv

def read_sensor_data_csv(filename):
    """Read sensor data from CSV file."""
    data = []
    
    try:
        with open(filename, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append({
                    'timestamp': row['timestamp'],
                    'sensor_id': row['sensor_id'],
                    'temperature': float(row['temperature']),
                    'humidity': float(row['humidity'])
                })
        
        print(f"Read {len(data)} sensor readings from CSV")
        return data
        
    except FileNotFoundError:
        print(f"CSV file '{filename}' not found")
        return []
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

def write_sensor_data_csv(data, filename):
    """Write sensor data to CSV file."""
    if not data:
        print("No data to write")
        return
    
    fieldnames = ['timestamp', 'sensor_id', 'temperature', 'humidity']
    
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Wrote {len(data)} records to '{filename}'")

# Create sample CSV data
sample_csv_data = [
    {'timestamp': '2023-10-01T10:00:00', 'sensor_id': 'TEMP01', 'temperature': 22.5, 'humidity': 65.0},
    {'timestamp': '2023-10-01T10:05:00', 'sensor_id': 'TEMP01', 'temperature': 23.1, 'humidity': 64.5},
    {'timestamp': '2023-10-01T10:10:00', 'sensor_id': 'TEMP01', 'temperature': 22.8, 'humidity': 65.2},
]

write_sensor_data_csv(sample_csv_data, 'sensor_data.csv')
csv_data = read_sensor_data_csv('sensor_data.csv')

# Analyze CSV data
temperatures_from_csv = [reading['temperature'] for reading in csv_data]
print(f"Average temperature from CSV: {sum(temperatures_from_csv)/len(temperatures_from_csv):.1f}°C")
```

## Object-Oriented Programming

### Classes and Objects

```python
class TemperatureSensor:
    """A class representing a temperature sensor."""
    
    # Class variable
    sensor_count = 0
    
    def __init__(self, sensor_id, location, calibration_offset=0.0):
        """Initialize a new temperature sensor."""
        self.sensor_id = sensor_id
        self.location = location
        self.calibration_offset = calibration_offset
        self.readings = []
        
        # Increment class counter
        TemperatureSensor.sensor_count += 1
        
        print(f"Created sensor {sensor_id} at {location}")
    
    def add_reading(self, temperature):
        """Add a temperature reading."""
        calibrated_temp = temperature + self.calibration_offset
        self.readings.append(calibrated_temp)
    
    def get_average_temperature(self):
        """Calculate average temperature."""
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)
    
    def get_reading_count(self):
        """Get number of readings."""
        return len(self.readings)
    
    def get_temperature_range(self):
        """Get min and max temperatures."""
        if not self.readings:
            return None, None
        return min(self.readings), max(self.readings)
    
    def print_report(self):
        """Print a summary report."""
        print(f"\nSensor Report: {self.sensor_id}")
        print(f"Location: {self.location}")
        print(f"Readings: {self.get_reading_count()}")
        
        if self.readings:
            avg = self.get_average_temperature()
            min_temp, max_temp = self.get_temperature_range()
            print(f"Average: {avg:.1f}°C")
            print(f"Range: {min_temp:.1f}°C to {max_temp:.1f}°C")
        else:
            print("No readings available")
    
    def __str__(self):
        """String representation of the sensor."""
        return f"TemperatureSensor({self.sensor_id}, {self.location})"
    
    def __repr__(self):
        """Developer representation of the sensor."""
        return f"TemperatureSensor('{self.sensor_id}', '{self.location}', {self.calibration_offset})"

# Create and use sensors
outdoor_sensor = TemperatureSensor("OUTDOOR-01", "Garden", -0.5)
indoor_sensor = TemperatureSensor("INDOOR-01", "Living Room")

# Add readings
outdoor_readings = [15.2, 16.8, 14.9, 17.1, 15.5]
indoor_readings = [22.1, 21.8, 22.5, 22.0, 21.9]

for temp in outdoor_readings:
    outdoor_sensor.add_reading(temp)

for temp in indoor_readings:
    indoor_sensor.add_reading(temp)

# Print reports
outdoor_sensor.print_report()
indoor_sensor.print_report()

print(f"\nTotal sensors created: {TemperatureSensor.sensor_count}")
```

### Inheritance

```python
class Sensor:
    """Base class for all sensors."""
    
    def __init__(self, sensor_id, location):
        self.sensor_id = sensor_id
        self.location = location
        self.readings = []
    
    def add_reading(self, value):
        """Add a reading (to be implemented by subclasses)."""
        self.readings.append(value)
    
    def get_reading_count(self):
        """Get number of readings."""
        return len(self.readings)
    
    def get_average(self):
        """Calculate average of readings."""
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)
    
    def print_basic_info(self):
        """Print basic sensor information."""
        print(f"Sensor: {self.sensor_id} at {self.location}")
        print(f"Readings: {self.get_reading_count()}")

class TemperatureSensor(Sensor):
    """Temperature sensor with calibration."""
    
    def __init__(self, sensor_id, location, calibration_offset=0.0):
        super().__init__(sensor_id, location)
        self.calibration_offset = calibration_offset
    
    def add_reading(self, temperature):
        """Add calibrated temperature reading."""
        calibrated_temp = temperature + self.calibration_offset
        super().add_reading(calibrated_temp)
    
    def get_unit(self):
        """Return temperature unit."""
        return "°C"
    
    def print_report(self):
        """Print detailed temperature report."""
        self.print_basic_info()
        if self.readings:
            avg = self.get_average()
            print(f"Average temperature: {avg:.1f}{self.get_unit()}")

class HumiditySensor(Sensor):
    """Humidity sensor."""
    
    def get_unit(self):
        """Return humidity unit."""
        return "%RH"
    
    def print_report(self):
        """Print detailed humidity report."""
        self.print_basic_info()
        if self.readings:
            avg = self.get_average()
            print(f"Average humidity: {avg:.1f}{self.get_unit()}")

# Create different sensor types
temp_sensor = TemperatureSensor("TEMP-01", "Office", -0.2)
humidity_sensor = HumiditySensor("HUM-01", "Office")

# Add readings
temp_sensor.add_reading(22.5)
temp_sensor.add_reading(23.1)
temp_sensor.add_reading(22.8)

humidity_sensor.add_reading(65.0)
humidity_sensor.add_reading(64.5)
humidity_sensor.add_reading(66.2)

# Print reports
temp_sensor.print_report()
print()
humidity_sensor.print_report()
```

## Error Handling

### Exception Handling

```python
def safe_calculate_statistics(data):
    """Calculate statistics with comprehensive error handling."""
    try:
        if not isinstance(data, (list, tuple)):
            raise TypeError("Data must be a list or tuple")
        
        if not data:
            raise ValueError("Cannot calculate statistics for empty dataset")
        
        # Check for non-numeric values
        numeric_data = []
        for i, value in enumerate(data):
            try:
                numeric_value = float(value)
                if not (float('-inf') < numeric_value < float('inf')):
                    raise ValueError(f"Invalid value at index {i}: {value}")
                numeric_data.append(numeric_value)
            except (ValueError, TypeError):
                print(f"Warning: Skipping non-numeric value at index {i}: {value}")
        
        if not numeric_data:
            raise ValueError("No valid numeric values found in dataset")
        
        # Calculate statistics
        n = len(numeric_data)
        mean = sum(numeric_data) / n
        variance = sum((x - mean) ** 2 for x in numeric_data) / n
        std_dev = variance ** 0.5
        
        return {
            'count': n,
            'mean': mean,
            'std_dev': std_dev,
            'min': min(numeric_data),
            'max': max(numeric_data)
        }
        
    except TypeError as e:
        print(f"Type error: {e}")
        return None
    except ValueError as e:
        print(f"Value error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Test error handling
test_cases = [
    [1, 2, 3, 4, 5],                    # Valid data
    [],                                  # Empty data
    [1, 2, "invalid", 4, 5],           # Mixed data types
    "not a list",                       # Wrong data type
    [float('inf'), 1, 2],               # Invalid numeric values
    None                                # None value
]

for i, test_data in enumerate(test_cases):
    print(f"\nTest case {i+1}: {test_data}")
    result = safe_calculate_statistics(test_data)
    if result:
        print(f"Statistics: {result}")
    else:
        print("Failed to calculate statistics")
```

### Custom Exceptions

```python
class DataProcessingError(Exception):
    """Base exception for data processing errors."""
    pass

class InvalidDataError(DataProcessingError):
    """Raised when data is invalid or corrupted."""
    pass

class InsufficientDataError(DataProcessingError):
    """Raised when there's not enough data for processing."""
    pass

class DataProcessor:
    """Data processor with custom exception handling."""
    
    def __init__(self, min_data_points=3):
        self.min_data_points = min_data_points
        self.data = []
    
    def add_data_point(self, value):
        """Add a data point with validation."""
        try:
            numeric_value = float(value)
            if not (-1000 <= numeric_value <= 1000):  # Reasonable range check
                raise InvalidDataError(f"Value {numeric_value} is outside valid range (-1000, 1000)")
            self.data.append(numeric_value)
        except (ValueError, TypeError):
            raise InvalidDataError(f"Cannot convert '{value}' to numeric value")
    
    def calculate_trend(self):
        """Calculate trend requiring minimum data points."""
        if len(self.data) < self.min_data_points:
            raise InsufficientDataError(
                f"Need at least {self.min_data_points} data points, have {len(self.data)}"
            )
        
        # Simple linear trend calculation
        n = len(self.data)
        x_values = list(range(n))
        
        # Calculate slope using least squares
        x_mean = sum(x_values) / n
        y_mean = sum(self.data) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, self.data))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope

# Usage with custom exceptions
processor = DataProcessor(min_data_points=5)

try:
    # Add various data points
    valid_data = [10.0, 10.5, 11.0, 11.2, 11.8]
    invalid_data = ["not_a_number", 2000, None]
    
    print("Adding valid data...")
    for value in valid_data:
        processor.add_data_point(value)
    
    print("Adding invalid data...")
    for value in invalid_data:
        try:
            processor.add_data_point(value)
        except InvalidDataError as e:
            print(f"  Rejected: {e}")
    
    print(f"\nCalculating trend with {len(processor.data)} data points...")
    trend = processor.calculate_trend()
    print(f"Trend slope: {trend:.4f}")

except InsufficientDataError as e:
    print(f"Cannot calculate trend: {e}")
except DataProcessingError as e:
    print(f"Data processing error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Working with External Libraries

### NumPy for Numerical Computing

```python
import numpy as np

# Create arrays
data = np.array([1.2, 2.3, 3.4, 4.5, 5.6])
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"Array: {data}")
print(f"Matrix:\n{matrix}")

# Array operations
squared = data ** 2
normalized = (data - np.mean(data)) / np.std(data)

print(f"Squared: {squared}")
print(f"Normalized: {normalized}")

# Statistical functions
print(f"Mean: {np.mean(data):.2f}")
print(f"Standard deviation: {np.std(data):.2f}")
print(f"Correlation matrix:\n{np.corrcoef(matrix)}")
```

### Pandas for Data Analysis

```python
import pandas as pd

# Create DataFrame
sensor_data = pd.DataFrame({
    'timestamp': pd.date_range('2023-10-01 10:00:00', periods=5, freq='5min'),
    'temperature': [22.5, 23.1, 22.8, 23.4, 22.9],
    'humidity': [65.0, 64.5, 66.2, 63.8, 65.1],
    'sensor_id': ['TEMP01'] * 5
})

print("Sensor Data:")
print(sensor_data)

# Data analysis
print(f"\nTemperature statistics:")
print(sensor_data['temperature'].describe())

# Data filtering
high_temp = sensor_data[sensor_data['temperature'] > 23.0]
print(f"\nHigh temperature readings:")
print(high_temp)
```

## Best Practices

### Code Organization

```python
# sensor_utils.py - Utility module
"""
Utility functions for sensor data processing.
"""

def validate_temperature(temp):
    """Validate temperature reading."""
    return -50 <= temp <= 100

def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

def calculate_heat_index(temp_c, humidity):
    """Calculate heat index given temperature and humidity."""
    temp_f = celsius_to_fahrenheit(temp_c)
    
    if temp_f < 80:
        return temp_c  # Heat index not applicable
    
    # Simplified heat index calculation
    hi = (temp_f + humidity) / 2  # This is simplified!
    return (hi - 32) * 5/9  # Convert back to Celsius

# main.py - Main application
"""
Main sensor data processing application.
"""

def main():
    """Main application entry point."""
    print("Sensor Data Processing Application")
    
    # Your main application logic here
    sensor = TemperatureSensor("MAIN-01", "Data Center")
    
    # Add some readings
    readings = [22.5, 23.1, 22.8, 23.4, 22.9]
    for reading in readings:
        sensor.add_reading(reading)
    
    sensor.print_report()

if __name__ == "__main__":
    main()
```

### Documentation and Type Hints

```python
from typing import List, Dict, Optional, Union

def process_sensor_readings(
    readings: List[float], 
    sensor_type: str = "temperature",
    calibration_offset: float = 0.0
) -> Dict[str, Union[float, int]]:
    """
    Process sensor readings and return statistics.
    
    Args:
        readings: List of sensor readings
        sensor_type: Type of sensor ('temperature', 'humidity', etc.)
        calibration_offset: Offset to apply to readings
        
    Returns:
        Dictionary containing statistical measures
        
    Raises:
        ValueError: If readings list is empty
        TypeError: If readings contain non-numeric values
        
    Example:
        >>> readings = [20.0, 21.0, 22.0]
        >>> stats = process_sensor_readings(readings)
        >>> print(stats['mean'])
        21.0
    """
    if not readings:
        raise ValueError("Readings list cannot be empty")
    
    # Apply calibration
    calibrated_readings = [r + calibration_offset for r in readings]
    
    return {
        'count': len(calibrated_readings),
        'mean': sum(calibrated_readings) / len(calibrated_readings),
        'min': min(calibrated_readings),
        'max': max(calibrated_readings),
        'sensor_type': sensor_type
    }

def find_sensor_by_id(
    sensors: List[TemperatureSensor], 
    sensor_id: str
) -> Optional[TemperatureSensor]:
    """
    Find a sensor by its ID.
    
    Args:
        sensors: List of sensor objects
        sensor_id: ID to search for
        
    Returns:
        Sensor object if found, None otherwise
    """
    for sensor in sensors:
        if sensor.sensor_id == sensor_id:
            return sensor
    return None
```

## Next Steps

This guide covers the essential Python concepts for data processing and visualization. Key areas to explore further:

1. **Scientific Computing:** NumPy, SciPy for advanced mathematical operations
2. **Data Analysis:** Pandas for data manipulation and analysis
3. **Visualization:** Matplotlib, Plotly, Seaborn for creating charts and plots
4. **Web Development:** Flask, Django for data dashboards
5. **Machine Learning:** Scikit-learn, TensorFlow for data modeling
6. **Testing:** unittest, pytest for code quality assurance

Python's ecosystem and ease of use make it an excellent choice for data processing tasks throughout this course!