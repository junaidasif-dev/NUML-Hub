import pandas as pd
import matplotlib.pyplot
data = pd.read_csv(r'E:\Sem3-GH\Sem3Lab\AI\knn\iris.csv')
species_counts = data['species'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Frequency of Species in Iris Dataset')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
