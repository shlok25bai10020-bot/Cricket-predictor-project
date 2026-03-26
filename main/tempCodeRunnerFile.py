import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import kagglehub
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')
class cricketPredictor:
    def download_dataset(self):
        print("---Downloading IPL dataset from dataset from Kaggle---")
        path=kagglehub.dataset_download("chaitu20/ipl-dataset2008-2025")
        print(f"Dataset downloaded to:{path}")
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        print(f"Available CSV files: {csv_files}")
    def setup_system(self, data_file="ipl_data.csv"):
        self.data_file = data_file
        self.isTrained = False
        self.encoders = {}
        self.model = None
        print(f"--- System Booting: Loading {self.data_file} ---")    
        if os.path.exists(self.data_file):
            self.train_model()
        else:
            print(f"Error: '{self.data_file}' not found in directory.")

    def cleanData(self, df):  #df stands for DataFrame
        if 'match_id' in df.columns:
            df = df.drop_duplicates(subset=['match_id'], keep='first').copy()
        df = df.rename(columns={
            'batting_team': 'team1',
            'bowling_team': 'team2',
            'match_won_by': 'winner'
        })
        teamNameChanges = {
            'Delhi Daredevils': 'Delhi Capitals',  #Name Change
            'Kings XI Punjab': 'Punjab Kings',     #Name Change
            'Deccan Chargers': 'Sunrisers Hyderabad', #Name Change
            'Pune Warriors': 'Dropped',               #Team Is Dropped
            'Gujarat Lions': 'Dropped',                #Team Is Dropped
            'Kochi Tuskers Kerala': 'Dropped',          #Team Is Dropped
            'Rising Pune Supergiant': 'Dropped',        #Team Is Dropped
            'Rising Pune Supergiants': 'Dropped'          #Team Is Dropped
        }
        df.replace({'team1': teamNameChanges, 'team2': teamNameChanges}, inplace=True)
        if 'team1' in df.columns and 'team2' in df.columns:
            df = df[(df['team1'] != 'Dropped') & (df['team2'] != 'Dropped')]
        needed = ['team1', 'team2', 'toss_decision', 'venue', 'winner']
        availableColumns = [col for col in needed if col in df.columns]
        if availableColumns:
            df = df.dropna(subset=availableColumns)
        return df

    def train_model(self):
        try:
            df = pd.read_csv(self.data_file)
            df = self.cleanData(df)
            features = ['team1', 'team2', 'toss_decision', 'venue']
            for col in features + ['winner']:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            X = df[features]
            y = df['winner']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model = RandomForestClassifier(
                n_estimators=150, 
                random_state=42, 
                max_depth=None, 
                class_weight='balanced'
            )
            self.model.fit(X_train, y_train)
            acc = self.model.score(X_test, y_test)
            print(f" Training Complete. Model Accuracy: {acc*100:.1f}%")
            importances = dict(zip(features, self.model.feature_importances_))
            print(f" Logic used (Feature Importance): {importances}\n")
            self.isTrained = True
        except Exception as e:
            print(f" Boot failed. Error: {e}")
            self.isTrained = False
    def predict_match(self):
        if not self.isTrained:
            print("System not ready.")
            return
        print("\n"  + " " + "MATCH PREDICTION ENGINE" + " " )
        print("-" * 35)
        try:
            validTeams = sorted(list(self.encoders['team1'].classes_))
            print(f"Available Teams: {', '.join(validTeams)}")
            
            t1 = input("\nEnter Team 1 (Batting First): ").strip()
            t2 = input("Enter Team 2 (Bowling First): ").strip()
            venue = input("Enter Venue: ").strip()
            tossD = input("Toss Decision (field/bat): ").strip()
            input_data = [
                self.encoders['team1'].transform([t1])[0],
                self.encoders['team2'].transform([t2])[0],
                self.encoders['toss_decision'].transform([tossD])[0],
                self.encoders['venue'].transform([venue])[0]
            ]
            prediction_encoded = self.model.predict([input_data])[0]
            winner_name = self.encoders['winner'].inverse_transform([prediction_encoded])[0]
            probs = self.model.predict_proba([input_data])[0]
            confidence = max(probs) * 100  
            print("\n" + "*"*40)
            print(f" PROJECTED WINNER: {winner_name}")
            print(f" PROBABILITY: {confidence:.1f}%")
            print("*"*40)
        except ValueError:
            print("\n  Input Error: Spelling must match 'Available Teams' list exactly.")
def main():
    workingModel = cricketPredictor()
    workingModel.setup_system("ipl_data.csv")

    while True:
        print("\n[1] Predict Match  [2] Exit")
        choice = input("Select: ")
        
        if choice == '1':
            workingModel.predict_match()
        elif choice == '2':
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()