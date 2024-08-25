import csv

ages = []
sexs = []
bmis = []
children = []
smokers = []
regions = []
charges = []

with open("insurance.csv", newline = '') as insurance_docs:
    reader = csv.DictReader(insurance_docs)
    for row in reader:
        ages.append(int(row['age']))
        sexs.append(row['sex'])
        bmis.append(float(row['bmi']))
        children.append(int(row['children']))
        smokers.append(row['smoker'])
        regions.append(row['region'])
        charges.append(float(row['charges']))

class Analysis:
    def __init__(self, ages, sexs, bmis, children, smokers, regions, charges):
        # Initalize attributes with data lists
        self.ages = ages
        self.sexs = sexs
        self.bmis = bmis
        self.children = children
        self.smokers = smokers
        self.regions = regions
        self.charges = charges

        # Initialize dictionaries to aggregate data
        self.sex_count = {'male': 0, 'female': 0}
        self.total_charges_per_sex = {'male': 0.0, 'female': 0.0}
        self.total_children_per_sex = {'male': 0, 'female': 0}
        self.total_charges_per_region = {'southwest': 0.0, 'southeast': 0.0, 'northwest': 0.0, 'northeast': 0.0}
        self.total_smokers_per_region = {'southwest': 0, 'southeast': 0, 'northwest': 0, 'northeast': 0}

        # Populate dictionaries with data
        for sex, charge, child, smoker, region in zip(sexs, charges, children, smokers, regions):
            # Update the count of males and females
            self.sex_count[sex] += 1
            # Update the total charges for the gender
            self.total_charges_per_sex[sex] += charge
            # Update the total number of children for the gender
            self.total_children_per_sex[sex] += child
            # Update the total charges for the region
            self.total_charges_per_region[region] += charge
            # If patient is a smoker, increment the smoker count for the region
            if smoker == 'yes':
                self.total_smokers_per_region[region] += 1
        
    def average_age(self):
        # Calculate average age of all patients, rounded to two decimal places
        return round(sum(self.ages) / len(self.ages), 2)
    
    def average_charge(self):
        # Calculate average charge of all patients, rounded to two decimal places
        return round(sum(self.charges) / len(self.charges), 2)
    
    def male_female_charges_average(self):
        # Calculate and return average charges for males and females
        return [
            round(self.total_charges_per_sex['male'] / self.sex_count['male'], 2),
            round(self.total_charges_per_sex['female'] / self.sex_count['female'], 2)
        ]
    
    def total_children_by_sex(self):
        # Return total amount of children for males and females
        return [ self.total_children_per_sex['male'], self.total_children_per_sex['female']]
    
    def total_pay_per_region(self):
        # Iterate through total_charges_per_region and returns total pay rounded in two decimals
        return {region: round(amount, 2) for region, amount in self.total_charges_per_region.items()}
    
    def male_female_counter(self):
        # Returns a dictionary with counts of males and females
        return self.sex_count

    def smokers_per_region(self):
        # Returns a dictionary with counts of smokers per region
        return self.total_smokers_per_region
    
    def total_pay_per_age_group(self):
        # Initializes a dictionary to group total charges by age ranges
        age_groups = {'19-': 0, '20-29': 0, '30-39': 0, '40-49': 0, '50-59': 0, '60+': 0}
        # Iterates over ages and accumulates them into age_groups dictionary
        for age, charge in zip(self.ages, self.charges):
            if age < 20: age_groups['19-'] += charge
            elif age < 30: age_groups['20-29'] += charge
            elif age < 40: age_groups['30-39'] += charge
            elif age < 50: age_groups['40-49'] += charge
            elif age < 60: age_groups['50-59'] += charge
            else: age_groups['60+'] += charge
        # Returns dictionary with age groups and total charges rounded to two decimals
        return {group: round(amount, 2) for group, amount in age_groups.items()}
    
analysis = Analysis(ages, sexs, bmis, children, smokers, regions, charges)

# Print results for verification
print("Average Age:", analysis.average_age())
print("Average Charge:", analysis.average_charge())
print("Average Charges by Sex:", analysis.male_female_charges_average())
print("Total Children by Sex:", analysis.total_children_by_sex())
print("Total Charges by Region:", analysis.total_pay_per_region())
print("Sex Distribution:", analysis.male_female_counter())
print("Smokers by Region:", analysis.smokers_per_region())
print("Charges by Age Group:", analysis.total_pay_per_age_group())

print(max(charges))