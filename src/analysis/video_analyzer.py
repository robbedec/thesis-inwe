import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import plotly.graph_objects as go
import sys

from scipy import stats, mean

from src.analysis.analyzer import StaticAnalyzer
from src.analysis.enums import Measurements, MEEIMovements
from src.utils.util import resize_with_aspectratio

def get_target_from_movement(movement: MEEIMovements):
    target = []

    if MEEIMovements.EYEBROWS == movement:
        target = [x.name for x in [
            Measurements.EYEBROW_EYE_DISTANCE,
            Measurements.EYEBROW_HORIZONTAL_DISTANCE,
            Measurements.EYEBROW_INTERCEPT_DISTANCE,
        ]]
    elif movement in [MEEIMovements.LIP_PUCKER, MEEIMovements.SMILE_OPEN, MEEIMovements.SMILE_CLOSED, MEEIMovements.BOTTOM_TEETH]:
        target = [x.name for x in [
            Measurements.MOUTH_AREA,
            Measurements.LIPCENTER_OFFSET,
        ]]
    elif movement in [MEEIMovements.EYE_CLOSE_HARD, MEEIMovements.EYE_CLOSE_SOFT]:
        target = [x.name for x in [
            Measurements.EYE_DROOP,
        ]]
    
    if len(target) == 0:
        raise Exception('No targets found for given movement.')
    
    other = [cat.name for cat in Measurements if cat.name not in target]

    return target, other


class VideoAnalyzer():
    def __init__(
            self,
            video_path,
            csv_path,
            rolling_window=0
        ):

        # Only used to fetch fps
        self._cap = cv2.VideoCapture(video_path)
        self._static_analyzer = StaticAnalyzer(mp_static_mode=False)
        self._csv_path = csv_path

        self._session_id = 0

        """
        if not os.path.exists(self._csv_path):
            # Create csv file with measurements if it does not exist
            self.process_video()
        else:
            self._df_results = pd.read_csv(self._csv_path, index_col=0)
            self._session_id = self._df_results['session_id'].nunique()
            
            if process_video:
                self.process_video(df_input=self._df_results)
        """
        self._df_results = pd.DataFrame() if not os.path.exists(self._csv_path) else pd.read_csv(self._csv_path, index_col=0)
        self._session_id = 0 if len(self._df_results) == 0 else self._df_results['session_id'].nunique()
        
        self._df_results_original = self._df_results.copy()

        # Apply a moving average to remove sudden changes between subsequent frames.
        # These can be attested to inaccuracies by the model.
        if rolling_window > 0:
            self._df_results = self._df_results.rolling(window=rolling_window).mean()
            # Rolling creates NaN values for the first (rolling_window - 1) entries
            # because it cant create an average.
            self._df_results = self._df_results.dropna()

        # Group results from frames together for every second
        # The resulting dataframe contains one aggregated entry for every second of the video.
        # TODO: This may be better suited in the audiogram function to preserve the raw overview function.
        # TODO: 1 second may be too fast to capture facial movement, consider e.g. 500ms.
        fps = self._cap.get(cv2.CAP_PROP_FPS) #/ 2
        #fps=1

        use_harmonic_mean = False
        if use_harmonic_mean:
            self._df_results = self._df_results.groupby(np.arange(len(self._df_results)) // fps).agg(lambda x: stats.hmean(x))
        else:
            self._df_results = self._df_results.groupby(np.arange(len(self._df_results)) // fps).mean()

    def process_video(self, movement: MEEIMovements, video_path):
        df_video = pd.DataFrame()
        cap = cv2.VideoCapture(video_path)

        if os.path.exists(self._csv_path):
            df_video = pd.read_csv(self._csv_path, index_col=0)

        while True:
            success, frame = cap.read()

            if not success:
                # End of video reached
                df_video.to_csv(self._csv_path)
                self._df_results = df_video

                break

            # Calculate scores for the frame
            try:
                self._static_analyzer.img = frame
                frame_results = self._static_analyzer.resting_symmetry()
            except:
                print("Not able to detect a face in the frame")
                continue

            df_video = df_video.append({
                **{ key.name: value for key, value in frame_results.items() },
                **{
                    'session_id': self._session_id,
                    'movement': movement,
                    # Maybe add date field
                }
            }, ignore_index=True)

    def display_frame(self, frames=[]):
        for frame_index in frames:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            _, frame = self._cap.read()

            cv2.imshow('frame', resize_with_aspectratio(frame, width=400))
            cv2.waitKey(0)
    
    def resume_video_from_frame(self, startframe=0):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, startframe)
        while True:
            success, frame = self._cap.read()

            if not success:
                break

            cv2.imshow('video', resize_with_aspectratio(frame, width=400))
            cv2.waitKey(1)

    def score_overview(self, single_plot=True, rolling_window=0):
        if self._csv_path is None:
            raise AttributeError('Provide the path to a valid csv file.')
        
        measurements_categories = [e_cat.name for e_cat in Measurements]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        data = self._df_results_original.rolling(window=rolling_window).mean() if rolling_window > 0 else self._df_results_original

        if single_plot:
            fig, axs = plt.subplots(len(measurements_categories))

        for i, cat in enumerate(measurements_categories):
            selected_color = colors[i % len(colors)]
            if single_plot:
                # Axis settings
                axs[i].plot(data[cat], c=selected_color)
                axs[i].set_title(cat, fontsize=8)
                axs[i].set_ylim([0,1])

                # Figure settings
                #fig.set_dpi(200)
                fig.tight_layout()
                fig.set_size_inches(18.5, 10.5)

                #plt.subplots_adjust(hspace=1.5)
                axs[i].axhline(y=0.5, color='black', linewidth=0.5, linestyle='--') # Draw midline

            else:
                plt.plot(data[cat], c=selected_color)
                plt.title(cat)
                plt.xlabel('frame number')
                plt.figure()

        plt.show()
    
    def audiogram(self, movement: MEEIMovements, session_id, video_path):
        target = []

        if MEEIMovements.EYEBROWS == movement:
            target = [x.name for x in [
                Measurements.EYEBROW_EYE_DISTANCE,
                Measurements.EYEBROW_HORIZONTAL_DISTANCE,
                Measurements.EYEBROW_INTERCEPT_DISTANCE,
            ]]
        elif MEEIMovements.LIP_PUCKER == movement:
            target = [x.name for x in [
                Measurements.MOUTH_AREA,
                Measurements.LIPCENTER_OFFSET,
            ]]

        synkinetic = [cat.name for cat in Measurements if cat.name not in target]

        # Create new dataframe with aggregated results
        df_movement = pd.DataFrame()
        for index, row in self._df_results[self._df_results['session_id'] == session_id].iterrows():
            # This variable will contain 2 values, the harmonic mean of the target
            # scores and that of the synkinetic scores.
            data = []
            for col_names in [target, synkinetic]:
                data.append(stats.hmean([row[cname] for cname in col_names]))

            df_movement = df_movement.append({ 'target': data[0], 'synkinetic': data[1] }, ignore_index=True)
            data.clear()
        
        # marker=mlp.markers.CARETDOWNBASE
        plt.plot(df_movement, marker='.')
        plt.legend(df_movement.columns, loc='best')
        plt.grid(b=True, which='major', color='#666666', linestyle='-')

        plt.xticks(np.arange(start=0, stop=len(df_movement), step=5))
        plt.yticks(np.arange(start=0, stop=1.25, step=0.25))

        plt.title('Movement: ' + movement.name)

        plt.show()

    def audiogramV2(self, movement: MEEIMovements, session_id):
        target = []

        target, synkinetic = get_target_from_movement(movement)

        # Create new dataframe with aggregated results
        df_movement = pd.DataFrame()
        for index, row in self._df_results[(self._df_results['session_id'] == session_id) & (self._df_results['movement'] == movement)].iterrows():
            target_agg = stats.hmean([row[cname] for cname in target])
            others = [row[cname] for cname in synkinetic]
            df_movement = df_movement.append({ 'target': target_agg, 'synkinetic': others }, ignore_index=True)

        """
        PLOT SCORES 
        """
        # Plot target line
        plt.plot(df_movement.iloc[:, 0], marker='.')

        # Loop over the other categories
        colors = ['#8c564b', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, x in enumerate(df_movement.iloc[:, 1]):
            plt.scatter(x=np.full(len(x), i), y=x, s=17, c=colors)
            #print(i, x)

        # Fix legend and its colors.
        plt.legend(['TARGET: ' + ', '.join(target)] + synkinetic, loc='best')
        ax = plt.gca()
        leg = ax.get_legend()
        for i in range(len(synkinetic)):
            leg.legendHandles[i + 1].set_color(colors[i])

        plt.grid(b=True, which='major', color='#666666', linestyle='-')

        plt.xticks(np.arange(start=0, stop=len(df_movement), step=5))
        plt.yticks(np.arange(start=0, stop=1.25, step=0.25))

        plt.title('Movement: ' + movement.name)

        plt.show()
    
    
    def radarplot(self, session_ids=[]):
        # https://plotly.com/python/radar-chart/
        # For different renderers https://plotly.com/python/renderers/

        # Or instead of .mean(), use .agg(lambda x: stats.hmean(x)) for harmonic mean.
        # TODO: Filter further so it only contains rows with the correct movement. This may be redundant because every row with the same session_id has the same measurement.
        df_data = self._df_results_original.loc[(self._df_results_original['session_id'].isin(session_ids)) | (len(session_ids) == 0)][[c.name for c in Measurements] + ['session_id']]
        #df_data = df_data.groupby('session_id').mean()
        df_data = df_data.groupby('session_id').agg(lambda x: stats.hmean(x))

        # TODO: Give better names
        # Append first item so the figure closes.
        categories = df_data.columns.to_list()
        categories.append(categories[0])

        # Init figure
        fig = go.Figure()

        for index, row in df_data.iterrows():
            dl = row.to_list()
            dl.append(dl[0])

            # Do in loop over rows of data is patient has been measured multiple times
            fig.add_trace(go.Scatterpolar(
                r=dl,
                theta=categories,
                #fill='toself',
                name='Meting ' + str(int(index + 1)),
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                )
            ),
            showlegend=True,
            title={
                'text': 'Movement: ' + self._movement.name,
                'xanchor': 'center',
                'yanchor': 'top', 
                'y':0.95,
                'x':0.48,
            },
        )

        fig.show()
    
    def sunnybrook_estimate(self, session_id):
        df_session = self._df_results[self._df_results['session_id'] == session_id]

        # TODO: include snarl
        sunnybrook_movements = [MEEIMovements.EYEBROWS, MEEIMovements.EYE_CLOSE_SOFT, MEEIMovements.SMILE_OPEN, MEEIMovements.LIP_PUCKER]

        resting_symmetry_score = sys.maxsize
        voluntary_movement_score = 0
        synkinesis_score = 0

        for movement in sunnybrook_movements:
            # Create new dataframe with aggregated results
            target, other = get_target_from_movement(movement)

            df_movement = pd.DataFrame()
            for index, row in df_session[df_session['movement'] == movement].iterrows():
                target_agg = stats.hmean([row[cname] for cname in target])
                others = [row[cname] for cname in other]
                df_movement = df_movement.append({ 'target': target_agg, 'synkinetic': others }, ignore_index=True)
            
            if len(df_movement) == 0:
                raise Exception('No information for {}'.format(movement.name))
            
            """
            ESTIMATE SCORE 
            """
            # Take scores from the first second to score resting symmetry
            selected_row = df_movement.iloc[0, :]

            # TODO: improve interval borders.
            intervals = [0.3, 0.55, 0.7, 0.8, 0.9, 1]
            fivepointscale = np.searchsorted(intervals, selected_row[1], side='right')

            # Group all info in 1 object:
            # Contains the Measurement name (key) and a tuple that contains the score and 5 point scale score (value).
            scores = dict(zip(other, zip(selected_row[1], fivepointscale)))
            #print(scores)

            # Compare eyes => 1 if wider or narrow, 0 if normal
            eye_metric = selected_row[0] if movement in \
                [MEEIMovements.EYE_CLOSE_SOFT, MEEIMovements.EYE_CLOSE_HARD] \
                else scores[Measurements.EYE_DROOP.name][0]
            rest_eye_score = 1 if eye_metric < 0.85 else 0

            rest_nasolabialfold_score = 2 - np.searchsorted([0.6, 0.9], scores[Measurements.NASOLABIAL_FOLD_SIGMA.name][0], side='right')
            
            mouth_metric = selected_row[0] if movement in \
                [MEEIMovements.LIP_PUCKER, MEEIMovements.SMILE_CLOSED, MEEIMovements.SMILE_OPEN, MEEIMovements.BOTTOM_TEETH] \
                else scores[Measurements.LIPCENTER_OFFSET.name][0]
            rest_mouth_score = 1 if mouth_metric < 0.85 else 0

            rest = 5 * (rest_eye_score + rest_nasolabialfold_score + rest_mouth_score)
            if rest < resting_symmetry_score:
                resting_symmetry_score = rest
            #print('Resting for {}: eye={}, wang={}, mond={}'.format(movement.name, rest_eye_score, rest_nasolabialfold_score, rest_mouth_score))

            # Find second with the lowest target value.
            # TODO: Average N lowest target values.
            # (if N is given and equals to the amount of times the movement is performed).
            target_col = df_movement.iloc[:, 0]
            lowest_target_indices = np.argsort(target_col)
            selected_row = df_movement.iloc[lowest_target_indices[0], :]

            voluntary_movement_score += np.searchsorted(intervals, selected_row[0], side='right') * 4
            synkinesis_score += round(sum([ tup[1] for tup in scores.values() ]) / len(other), 0)
            print('Movement {}: score={}, synk:{}, raw={}'.format(movement.name, np.searchsorted(intervals, selected_row[0], side='right'), round(sum([ tup[1] for tup in scores.values() ]) / len(other), 0), selected_row[0]))

        composite_score = voluntary_movement_score - round(resting_symmetry_score / len(sunnybrook_movements), 0) - synkinesis_score
        print('Voluntary movement score: {}\nResting symmetry score: {}\nSynkinesis score: {}\nComposite score: {}'.format(voluntary_movement_score, resting_symmetry_score, synkinesis_score, composite_score))


def generate_master_file():
    # Create one file with all movements from a single session
    csv_path = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/patient-master-1.csv'
    movs = [
        MEEIMovements.EYEBROWS,
        MEEIMovements.EYE_CLOSE_SOFT,
        #MEEIMovements.SNARL,
        MEEIMovements.SMILE_OPEN,
        MEEIMovements.LIP_PUCKER,
    ]

    # Master file 1
    video_paths1 = [
        '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20160601 (1).MPG',
        '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20160601 (2).MPG',
        #SNARL'/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20160601 (3).MPG',
        '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20160601 (4).MPG',
        '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20160601 (5).MPG',
    ]
    
    # Master file 2
    video_paths2 = [
        '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20170317 (1).MPG',
        '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20170317 (2).MPG',
        #SNARL'/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20170317 (6).MPG',
        '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20170317 (3).MPG',
        '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20170317 (7).MPG',
    ]

    # Create analyzer twice to update session_id
    videoanalyzer = VideoAnalyzer(video_path=video_path, csv_path=csv_path)
    [ videoanalyzer.process_video(movement=mov, video_path=vpath) for mov, vpath in zip(movs, video_paths1) ]
    videoanalyzer = VideoAnalyzer(video_path=video_path, csv_path=csv_path)
    [ videoanalyzer.process_video(movement=mov, video_path=vpath) for mov, vpath in zip(movs, video_paths2) ]

def sunnybrook_test():
    # Enkel gebruikt voor fps
    video_path = '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20180812 (5).MPG'
    csv_path = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/patient-master-1.csv'

    videoanalyzer = VideoAnalyzer(video_path=video_path, csv_path=csv_path)

    print('\n----------------------------------------')
    print('SUNNYBROOK TEST 1 (20160601): ')
    print('----------------------------------------')
    videoanalyzer.sunnybrook_estimate(session_id=0)

    """
    REMARKS:
    - Bewegingsscore bij de wenkbrauw is fout omdat de wenkbrauw niet juist wordt gedetecteerd. Die is slechts
      heel licht zichtbaar bij de mevrouw. 

    - Bewegingsscore bij zacht ogen sluiten krijgt het een lage score omdat het ene oog sneller dicht gaat dan de andere
      (blijft precies wat hangen) waardoor het laagste punt van de target niet overeen komt met het hoogtepunt van de 
      beweging.
    
    """
    print('\n----------------------------------------')
    print('SUNNYBROOK TEST 2 (20170317): ')
    print('----------------------------------------')
    videoanalyzer.sunnybrook_estimate(session_id=1)

if __name__ == '__main__':
    # Video volledige verlamming
    video_path = '/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/CompleteFlaccid/CompleteFlaccid1/CompleteFlaccid1.mp4'
    csv_path = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/CompleteFlaccid1.csv'

    # Video normal
    # video_path = '/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Normals/Normal8/Normal8.mp4'
    # csv_path = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/Normal8.csv'

    # Tuiten van de lippen, voor en na therapie.
    # video_path = '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20160601 (5).MPG'
    #video_path = '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20160711 (5).MPG'
    video_path = '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20180812 (5).MPG'
    csv_path = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/patient1.csv'
    
    
    csv_path = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/patient-master-1.csv'

    videoanalyzer = VideoAnalyzer(video_path=video_path, csv_path=csv_path)
    #videoanalyzer.score_overview(rolling_window=10)
    videoanalyzer.audiogramV2(movement=MEEIMovements.LIP_PUCKER, session_id=1)
    #videoanalyzer.radarplot(session_ids=[])
    sunnybrook_test()