import pandas as pd
import cv2
import matplotlib.pyplot as plt

from src.analysis.analyzer import StaticAnalyzer
from src.analysis.enums import Measurements
from src.utils.util import resize_with_aspectratio

class VideoAnalyzer():
    def __init__(self, video_path, csv_path=None):
        self._cap = cv2.VideoCapture(video_path)
        self._static_analyzer = StaticAnalyzer(mp_static_mode=False)
        self._csv_path = csv_path

    def process_video(self):
        # Prepare dataframe
        df_video = pd.DataFrame()

        while True:
            success, frame = self._cap.read()

            if not success:
                # End of video reached
                df_video.to_csv(self._csv_path if csv_path is not None else './test_normal.csv')
                break

            self._static_analyzer.img = frame

            frame_results = self._static_analyzer.resting_symmetry()

            df_video = df_video.append({ key.name: value for key, value in frame_results.items()}, ignore_index=True) 
    
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


    def results(self, single_plot=True):
        if self._csv_path is None:
            raise AttributeError('Provide the path to a valid csv file.')
        
        df_results = pd.read_csv(self._csv_path)

        measurements_categories = [e_cat.name for e_cat in Measurements]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        if single_plot:
            fig, axs = plt.subplots(len(measurements_categories))

        for i, cat in enumerate(measurements_categories):
            selected_color = colors[i % len(colors)]
            if single_plot:
                axs[i].plot(df_results[cat], c=selected_color)
                axs[i].set_title(cat, fontsize=8)
                #plt.subplots_adjust(hspace=1.5)
                #fig.set_dpi(200)
                fig.tight_layout()
                fig.set_size_inches(18.5, 10.5)

            else:
                plt.plot(df_results[cat], c=selected_color)
                plt.title(cat)
                plt.xlabel('frame number')
                plt.figure()

        plt.show()

if __name__ == '__main__':
    video_path = '/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/CompleteFlaccid/CompleteFlaccid1/CompleteFlaccid1.mp4'
    csv_path = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/test_video.csv'

    #video_path = '/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Normals/Normal8/Normal8.mp4'
    #csv_path = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/test_video_normal.csv'

    videoanalyzer = VideoAnalyzer(video_path=video_path, csv_path=csv_path)

    #videoanalyzer.process_video()
    videoanalyzer.results()
    #videoanalyzer.display_frame([1199, 2531])

    # Video that shows mouth movement
    #videoanalyzer.resume_video_from_frame(1400)
    #videoanalyzer.resume_video_from_frame(900)