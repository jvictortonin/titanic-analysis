import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetic parameters for seaborn
sns.set(style="whitegrid")  
plt.rcParams['figure.figsize'] = (8, 5)

# Load the Titanic dataset
df=pd.read_csv('train.csv')
print(df.head())

#Initial Data Exploration
print("Dimension:", df.shape)
print("Info", df.info())
print("Missing Values:\n", df.isnull().sum())
print("Statistical Summary:\n", df.describe())  

# Data Cleaning
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Survived"] = df["Survived"].astype(int)
#df["Pclass"] = df["Pclass"].astype("category")
#df["Sex"] = df["Sex"].astype("category")

#Quick analysis 
survivals = df["Survived"].mean()*100
print(f"Survival Rate: {survivals:.2f}%")   
df.groupby("Sex")["Survived"].mean()*100
df.groupby("Pclass")["Survived"].mean()*100
surv_pivot = pd.pivot_table(df, values="Survived", index="Sex", columns="Pclass",
                             aggfunc="mean")*100
print(surv_pivot)
count_pivot = pd.pivot_table(df, values="PassengerId", index="Sex", columns="Pclass",
                             aggfunc="count")*100
print(count_pivot)


# Visualizations
sns.barplot(x="Sex", y="Survived", data=df)
plt.title("Survival Rate x Sex")
plt.savefig("survival_rate_x_sex.png")
plt.close()

sns.barplot(x="Pclass", y="Survived", data=df)
plt.title("Survival Rate x Pclass") 
plt.savefig("survival_rate_x_Pclass.png")
plt.close()

sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")           
plt.savefig("age_distribution.png")
plt.close()

plt.figure(figsize=(6,4))
sns.heatmap(surv_pivot, annot=True, fmt=".1f", cmap="Blues")
plt.title("Survival Rate(%) per sex and class")
plt.ylabel("Sex")
plt.xlabel("Class")
plt.savefig("hetmap_survival_rate.png")
plt.close()

plt.figure(figsize=(6,4))
sns.heatmap(count_pivot, annot=True, fmt="d", cmap="Greens")
plt.title("Count of Passengers per sex and class")
plt.ylabel("Sex")
plt.xlabel("Class")
plt.savefig("hetmap_count_passengers.png")
plt.close()

#Some conclusions

def generate_report(df, export_txt = True, filename = "titanic_report.txt"):
    report = []

    mean_age = df["Age"].mean()
    median_age = df["Age"].median()

    report.append(f"The average age of passengers was about {mean_age:.1f} years,"
                    f"with a median age of {median_age:.1f} years.\n")

    surv_by_sex = df.groupby("Sex")["Survived"].mean()*100
    report.append(f"By gender, {surv_by_sex['female']:.1f}% of women survived "
    f"compared to only {surv_by_sex['male']:.1f}% of men. \n")

    surv_by_class = df.groupby("Pclass")["Survived"].mean()*100
    report.append(f"By class, survival rates were {surv_by_class[1]:.1f}% for 1st class,"
                   f"{surv_by_class[2]:.1f}% for 2nd class, and {surv_by_class[3]:.1f}% for 3rd class.\n")

    max_survival = surv_pivot.max().max()
    min_survival = surv_pivot.min().min()
    max_pos = surv_pivot.stack().idxmax()
    min_pos = surv_pivot.stack().idxmin()

    report.append(f"The highest survival rate was {max_survival:.1f}% for {max_pos[0]}s in {max_pos[1]} class,"
                   f"while the lowest was {min_survival:.1f}% for {min_pos[0]}s in {min_pos[1]} class.\n")    

    total_by_sex = count_pivot.sum(axis=1)
    most_passengers = count_pivot.sum(axis=0).idxmax()
    report.append(f"In terms of passenger distribution, there were {total_by_sex['male']} men"
                             f" and {total_by_sex['female']} women.\n")
    

    if export_txt:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("== FINAL REPORT ==\n\n")
            for item in report:
                f.write("• "+item+ "\n")
            f.write("\n==================\n")
        print(f"Report exported to {filename}")

    return report

final_report = generate_report(df)

print("\n== FINAL REPORT ==\n")
for item in final_report:
    print("• "+item)
print("\n==================\n")


 












