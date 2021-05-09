import csv
# Given a csv file, this method returns team stats as a 2D array and labels that denote
# whether or not a team has reached the playoffs
def extract_data(filename):
    # Initialize return values
    team_data_west = []
    team_data_east = []
    west_labels = {}
    east_labels = {}
    
    # Open csv file
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile)
        
        # Extract year
        _ = filename[:4]
        
        # Skip csv data table header
        next(readCSV)
        
        # Transform data into 2D array
        data = list(readCSV)
        
        # Remove league averages from data
        data.pop(len(data)-1)
        
        # Loop through the data
        for i in range(0, len(data)):
            # Remove irrelevant data
            data[i].pop(5)                # MP
            data[i].pop(4)                # G
            conference = data[i].pop(3)   # Conference
            record = data[i].pop(2)       # Record
            record = record[:2]
            if record[1] == '-':
                record = record[0]
            team_name = data[i].pop(1)    # Team
            if team_name[-1] == '*':
                team_name = team_name[:-1]
            data[i].pop(0)                # Rk
            
            # Add to team data dictionaries and labels
            if conference == "West":
                features = [float(stat) for stat in data[i]]
                features.insert(0, team_name)
                team_data_west.append(features)
                west_labels[team_name] = int(record)
            else:
                features = [float(stat) for stat in data[i]]
                features.insert(0, team_name)
                team_data_east.append(features)
                east_labels[team_name] = int(record)
        
        # Update ranking based on conference and record
        new_w = {}
        new_e = {}
        
        west_counter = 1
        sorted_west = sorted(west_labels.items(), key=lambda x: x[1], reverse=True)
        for team in sorted_west:
            new_w[team[0]] = west_counter
            west_counter += 1
            
        east_counter = 1
        sorted_east = sorted(east_labels.items(), key=lambda x: x[1], reverse=True)
        for team in sorted_east:
            new_e[team[0]] = east_counter
            east_counter += 1

    return team_data_west, team_data_east, new_w, new_e

# Get year data from start_year to end_year
def get_year_data(start_year, end_year):
    # Extract data from csv files from start_year to end_year
    first_year = start_year
    last_year = end_year

    # Store team stats into X and labels into y
    west_year_data = {}
    east_year_data = {}
    y_west = {}
    y_east = {}

    data_dir = '../data/'
    for i in range(first_year, last_year + 1):
        filename = data_dir + str(i) + '.csv'
        west_stats, east_stats, west_seeding, east_seeding = extract_data(filename)
        west_year_data[i] = west_stats
        east_year_data[i] = east_stats
        y_west[i] = west_seeding
        y_east[i] = east_seeding
        
    return west_year_data, east_year_data, y_west, y_east