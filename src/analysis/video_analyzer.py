import pandas as pd
import cv2
import matplotlib as mlp
import matplotlib.pyplot as plt
import os
import numpy as np
import plotly.graph_objects as go

from scipy import stats

from src.analysis.analyzer import StaticAnalyzer
from src.analysis.enums import Measurements, MEEIMovements
from src.utils.util import resize_with_aspectratio


class VideoAnalyzer():
    def __init__(
            self,
            movement: MEEIMovements,
            video_path,
            csv_path='./video_analysis_results.csv',
            process_video=True,
            rolling_window=0
        ):

        self._cap = cv2.VideoCapture(video_path)
        self._static_analyzer = StaticAnalyzer(mp_static_mode=False)
        self._csv_path = csv_path

        # Check if movement exist in Enum
        self._movement = movement

        if not os.path.exists(self._csv_path):
            # Create csv file with measurements if it does not exist
            self.process_video()
        else:
            self._df_results = pd.read_csv(self._csv_path, index_col=0)
            
            if process_video:
                self.process_video(df_input=self._df_results)
        
        self._df_results_original = self._df_results.copy()

        # Apply a moving average to remove sudden changes between subsequent frames.
        # These can be attested to inaccuracies by the model.
        #if rolling_window > 0:
        #    self._df_results = self._df_results.rolling(window=rolling_window).mean()
        #    # Rolling creates NaN values for the first (rolling_window - 1) entries
        #    # because it cant create an average.
        #    self._df_results = self._df_results.dropna()

        # Group results from frames together for every second
        # The resulting dataframe contains one aggregated entry for every second of the video.
        # TODO: This may be better suited in the audiogram function to preserve the raw overview function.
        # TODO: 1 second may be too fast to capture facial movement, consider e.g. 500ms.
        fps = self._cap.get(cv2.CAP_PROP_FPS)

        use_harmonic_mean = False
        if use_harmonic_mean:
            self._df_results = self._df_results.groupby(np.arange(len(self._df_results)) // fps).agg(lambda x: stats.hmean(x))
        else:
            self._df_results = self._df_results.groupby(np.arange(len(self._df_results)) // fps).mean()


    def process_video(self, df_input=pd.DataFrame()):
        # Prepare dataframe
        df_video = df_input

        session_id = 0 if len(df_video.index) == 0 else df_video['video_id'].nunique()

        while True:
            success, frame = self._cap.read()

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
                    'video_id': session_id,
                    'movement': self._movement,
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

    def score_overview(self, single_plot=True):
        if self._csv_path is None:
            raise AttributeError('Provide the path to a valid csv file.')
        
        measurements_categories = [e_cat.name for e_cat in Measurements]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        if single_plot:
            fig, axs = plt.subplots(len(measurements_categories))

        for i, cat in enumerate(measurements_categories):
            selected_color = colors[i % len(colors)]
            if single_plot:
                # Axis settings
                axs[i].plot(self._df_results[cat], c=selected_color)
                axs[i].set_title(cat, fontsize=8)
                axs[i].set_ylim([0,1])

                # Figure settings
                #fig.set_dpi(200)
                fig.tight_layout()
                fig.set_size_inches(18.5, 10.5)

                #plt.subplots_adjust(hspace=1.5)
                axs[i].axhline(y=0.5, color='black', linewidth=0.5) # Draw midline

            else:
                plt.plot(self._df_results[cat], c=selected_color)
                plt.title(cat)
                plt.xlabel('frame number')
                plt.figure()

        plt.show()
    
    def audiogram(self, movement: MEEIMovements):
        target = []

        if MEEIMovements.EYEBROWS == movement:
            target = [x.name for x in [
                Measurements.EYEBROW_EYE_DISTANCE,
                Measurements.EYEBROW_HORIZONTAL_DISTANCE,
                Measurements.EYEBROW_INTERCEPT_DISTANCE,
            ]]

        synkinetic = [cat.name for cat in Measurements if cat.name not in target]

        # Create new dataframe with aggregated results
        df_movement = pd.DataFrame()
        for index, row in self._df_results.iterrows():
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
    
    
    def radarplot(self, video_ids):
        # https://plotly.com/python/radar-chart/
        # For different renderers https://plotly.com/python/renderers/

        # Or instead of .mean(), use .agg(lambda x: stats.hmean(x)) for harmonic mean.
        df_data = self._df_results.loc[self._df_results['video_id'].isin(video_ids)][[c.name for c in Measurements] + ['video_id']]
        df_data = df_data.groupby('video_id').mean()
        # df_data = df_data.groupby('video_id').agg(lambda x: stats.hmean(x))

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
                name='Meting ' + str(index + 1),
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                )
            ),
            showlegend=False,
        )

        fig.show()

if __name__ == '__main__':
    video_path = '/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/CompleteFlaccid/CompleteFlaccid1/CompleteFlaccid1.mp4'

    video_path = '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20160601 (1).MPG'
    video_path = '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20160914 (6) oefening.MPG'
    video_path = '/media/robbedec/BACKUP/ugent/master/masterproef/data/patienten_liesbet/20160601 (3).MPG'

    csv_path = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/patient1.csv'

    #video_path = '/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Normals/Normal8/Normal8.mp4'
    #csv_path = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/test_video_normal.csv'

    videoanalyzer = VideoAnalyzer(movement=MEEIMovements.EYEBROWS, video_path=video_path, csv_path=csv_path, process_video=True)

    #videoanalyzer.process_video()
    #videoanalyzer.score_overview()
    #videoanalyzer.display_frame([1199, 2531])
    #videoanalyzer.audiogram(movement=MEEIMovements.EYEBROWS)
    videoanalyzer.radarplot(video_ids=[0, 1, 2])

    # Video that shows mouth movement
    #videoanalyzer.resume_video_from_frame(1400)
    #videoanalyzer.resume_video_from_frame(900)