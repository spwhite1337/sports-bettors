import os
import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

from sports_bettors.analytics.model.model import Model


class Validate(Model):

    @staticmethod
    def _baseline_prob():
        return 0.541

    def validate(self):
        df_, df_val, df = self.fit_transform()

        df_['preds'] = self.predict_spread(df_)
        df_val['preds'] = self.predict_spread(df_val)
        df['preds'] = self.predict_spread(df)

        # Preds vs Response
        with PdfPages(os.path.join(self.save_dir, 'validate_spreads.pdf')) as pdf:
            bins = np.linspace(-50, 50, 21)
            plt.figure()
            plt.hist(df_val['preds'], alpha=0.5, label='preds', bins=bins)
            plt.hist(df_val[self.response], alpha=0.5, label='Actuals', bins=bins)
            plt.xlabel(self.response)
            plt.title('Preds Distribution')
            plt.grid(True)
            plt.legend()
            pdf.savefig()
            plt.close()

            df_val['res'] = df_val['preds'] - df_val[self.response]
            df_['res'] = df_['preds'] - df_[self.response]
            plt.figure()
            plt.hist(df_val['res'], alpha=0.5, label='test', density=True, bins=bins)
            plt.hist(df_['res'], alpha=0.5, label='train', density=True, bins=bins)
            plt.title('Residuals Distribution')
            plt.grid(True)
            plt.legend()
            pdf.savefig()
            plt.close()

            # Heteroskedasticity
            plt.figure()
            plt.scatter(df_val['preds'], df_val['res'], label='residuals')
            plt.grid(True)
            plt.xlabel('Predictions')
            plt.ylabel('Residuals')
            plt.title('Heteroskedasticity')
            plt.legend()
            pdf.savefig()
            plt.close()

            # Compared to spread
            plt.figure()
            plt.hist(df_val['res'], alpha=0.5, label='residuals', bins=bins)
            plt.hist(df_val['spread_diff'], alpha=0.5, label='spread-diff', bins=bins)
            plt.grid(True)
            plt.legend()
            plt.title('Model vs. Spread')
            plt.xlabel('Error')
            pdf.savefig()
            plt.close()

            # As classifier
            classifer_response = 'away_team_wins_ats'
            # Logic:
            # If preds (predicted number of points to add to away team to get a tie)
            # is less than spread-line, then we would predict the away team to cover
            # e.g. spread is -7 but we predict -9, or spread is 5 and we predict -3
            # Therefore, spread-line minus preds is positively correlated with away team covering
            df['preds_c'] = df['spread_line'] - df['preds']
            # If the actual number of points to add for the away team to get a tie is more than the spread
            # e.g. -5 vs -7 then they lost
            # If the listed spread was greater than observed, then the away team won
            # e.g. -5 vs. -7 or +5 vs +2
            df[classifer_response] = (df['spread_line'] > df['spread_actual']).astype(int)
            df_ = df[df['gameday'] < (pd.Timestamp(self.TODAY) - pd.Timedelta(days=self.val_window))].copy()
            df_val = df[df['gameday'] > (pd.Timestamp(self.TODAY) - pd.Timedelta(days=self.val_window))].copy()

            # ROC
            fpr, tpr, thresholds = roc_curve(df_[classifer_response], df_['preds_c'])
            auc = roc_auc_score(df_[classifer_response], df_['preds_c'])
            plt.figure()
            plt.plot(fpr, tpr)
            plt.text(0.2, 0.9, f'AUC: {auc:.3f}\nn={df_val.shape[0]}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title('Train')
            plt.grid(True)
            pdf.savefig()
            plt.close()

            fpr, tpr, thresholds = roc_curve(df_val[classifer_response], df_val['preds_c'])
            auc = roc_auc_score(df_val[classifer_response], df_val['preds_c'])
            plt.figure()
            plt.plot(fpr, tpr)
            plt.text(0.2, 0.9, f'AUC: {auc:.3f}\nn={df_val.shape[0]}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.grid(True)
            plt.title('Test')
            pdf.savefig()
            plt.close()

            precision, recall, thresholds = precision_recall_curve(df_[classifer_response], df_['preds_c'])
            plt.figure()
            plt.plot(thresholds, precision[1:], label='precision')
            plt.plot(thresholds, recall[1:], label='recall')
            plt.hlines(df_val[classifer_response].mean(), min(thresholds), max(thresholds), color='black')
            plt.hlines(0.525, min(thresholds), max(thresholds), color='gray')
            plt.legend()
            plt.grid(True)
            plt.title('Train')
            pdf.savefig()
            plt.close()

            precision, recall, thresholds = precision_recall_curve(df_val[classifer_response], df_val['preds_c'])
            plt.figure()
            plt.plot(thresholds, precision[1:], label='precision')
            plt.plot(thresholds, recall[1:], label='recall')
            plt.legend()
            plt.hlines(df_val[classifer_response].mean(), min(thresholds), max(thresholds), color='black')
            plt.hlines(0.525, min(thresholds), max(thresholds), color='gray')
            plt.grid(True)
            plt.title('Test')
            pdf.savefig()
            plt.close()

            # Win-rate
            records = []
            for threshold in np.linspace(-10, 10, 41):
                fraction = df_val[df_val['preds_c'] >= threshold].shape[0] / df_val.shape[0]
                wins = df_val[(df_val['preds_c'] >= threshold)].shape[0] * \
                       df_val[(df_val['preds_c']) >= threshold][classifer_response].mean()
                total = df_val[(df_val['preds_c']) >= threshold].shape[0]
                record = {
                    'threshold': threshold,
                    'fraction': fraction,
                    'wins': wins,
                    'total': total,
                    'win_rate': wins / total if total > 0 else None
                }
                records.append(record)
            df_plot = pd.DataFrame.from_records(records)
            baseline_prob = self._baseline_prob()
            df_plot['win_rate_total'] = baseline_prob * (1 - df_plot['fraction']) + \
                                        df_plot['win_rate'] * df_plot['fraction']
            plt.figure()
            plt.plot(df_plot['threshold'], df_plot['win_rate'], label='win-rate')
            plt.plot(df_plot['threshold'], df_plot['win_rate_total'], label='win_rate_total')
            plt.legend()
            plt.ylabel('win-rate')
            plt.xlabel('Predicted Spread - Initial Spread')
            plt.hlines(0.525, -5, 5, color='black')
            plt.hlines(1-0.525, -5, 5, color='black')
            plt.hlines(baseline_prob, -5, 5, 'gray')
            plt.hlines(1-baseline_prob, -5, 5, 'gray')
            plt.ylim([0.4, 0.75])
            plt.grid(True)
            pdf.savefig()
            plt.close()

    def save_results(self):
        import pickle

    def predict_next_week(self) -> pd.DataFrame:
        df = pd.read_csv(self.link_to_data, parse_dates=['gameday'])
        df = self.engineer_features(df)
        df_ = df[
            df['gameday'].between(pd.Timestamp(self.TODAY), pd.Timestamp(self.TODAY) + datetime.timedelta(days=10))
            |
            # Keep this SF game as a test case
            (df['game_id'] == '2023_07_SF_MIN')
        ].copy()
        df_ = df_[(~df_['money_line'].isna() & ~df_['spread_line'].isna()) | (df_['game_id'] == '2023_07_SF_MIN')]

        df_['preds'] = self.predict_spread(df_)
        df_['preds_c'] = df_['spread_line'] - df_['preds']
        print(df_)
        df_.to_csv(os.path.join(os.getcwd(), 'data', 'df_test.csv'), index=False)
        return df_
