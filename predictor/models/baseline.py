#!/usr/bin/env python
# coding: utf-8

import datastore.datastore as datastore
import plotly.graph_objects as go
import tensorflow as tf
import sklearn.metrics
import pandas as pd
import numpy as np
import threading
import datetime
import queue
import json
import time
import sys
import sti

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

class BaselineOracle:
    """
    A class that forecasts Bitcoin prices one minute into the future in real time.
    """
    
    _warmed_up = False
    
    def __init__(self
        , look_back=60
        , days_for_training=1
        , batch_size=5
        , epochs=10
        , refresh_rate=3
        , leniency=5
        , max_retries=3):
        """
        Parameter(s):
        -------------
            look_back : int
                The number of past minutes to use for the next prediction.
            
            days_for_training : int
                The number of days to train the model on.
            
            batch_size : int
                The batch size used for training.
            
            epochs : int
                The number of training epochs.
                
            refresh_rate : int
                The minimum number of minutes that must occur before the model is 
                retrained.
                
            leniency : int
                We reject a model if its training loss is greater than:
                
                    MSE(prices[1:], prices[:-1]) + abs(leniency) * standard_deviation(prices)
                
                Thus, leniency controls how strict of a standard we have on model training.
                If we're too strict, we'll never find a model, but if we're too lenient, we
                end up with a poor performing model.
                        
            max_retries : int
                The maximum number of times the model should be trained and evaluated. If
                too many retries are specified and the leniency is too low, training will
                take too long and the model will not be trained on any new data that comes
                in during this time.
                
        Notes:
        ------        
            What happens if a new model is in the middle of training AND it's time to refresh?
                The program will continue to train the new model. If the new model is 
                sufficient, it will be used for predictions and the the refresh event 
                will be canceled. If a model is not found, the refresh event starts.
        """
        if max_retries       < 1: raise ValueError('Model must be trained at least once.')
        if days_for_training < 1: raise ValueError('Model must be trained on some data.')
        if refresh_rate      < 0: raise ValueError('Refresh rate must be 0 or positive.')
        self._datastores = datastore.DataStore(None, realtime_prices=True)
        self._lstmoracle = None
        self._loopthread = None
        self._lstmthread = None
        self._btc_actual = None
        self._history_df = None
        self._is_running = False
        self._projection = []
        self._proj_times = []
        self._lstm_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._is_updated = threading.Event()
        self._has_predic = threading.Event()
        self._look_backp = look_back
        self._train_days = days_for_training
        self._batch_size = batch_size
        self._num_epochs = epochs
        self._rfrsh_rate = refresh_rate
        self._num_stddev = leniency
        self._retr_limit = max_retries
        
        if not BaselineOracle._warmed_up:
            time.sleep(60 - datetime.datetime.now().second)
            BaselineOracle._warmed_up = True
    
    def _log_msg(self, msg):
        time_stamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")
        print("({}): {}".format(time_stamp, msg))
    
    def _create_model(self, neurons=10, opt='Nadam'):
        model = Sequential()
        model.add(
            LSTM(neurons,
                activation='relu',
                input_shape=(self._look_backp, 1)
            )
        )
        model.add(Dense(1))
        model.compile(optimizer=opt, loss='mse')
        return model
    
    def _retrain(self, prices, show_progress, verbose):
        
        # Convert data into an appropriate format
        gen = TimeseriesGenerator(prices.reshape((-1, 1)),
                                  prices.reshape((-1, 1)),
                                  length=self._look_backp, 
                                  batch_size=self._batch_size)
        
        # Compute the loss ceiling
        curr_mse = sklearn.metrics.mean_squared_error(prices[1:], prices[:-1])
        max_loss = curr_mse + abs(self._num_stddev) * np.std(prices)
        
        # Retrain until we run out of tries
        best_loss, best_model, retry_count, = float('inf'), None, 0
        if verbose: self._log_msg("Training new model..."); self._log_msg("Max Loss Tolerated: {}s".format(max_loss))
        while retry_count < self._retr_limit:
            new_model = self._create_model()
            new_model.fit(gen, epochs=self._num_epochs, shuffle=False, verbose=show_progress)
            curr_loss = new_model.evaluate(gen, verbose=show_progress)
            if curr_loss < best_loss:
                best_model = new_model
                best_loss  = curr_loss
            if verbose: self._log_msg('Generation {} loss: {}'.format(retry_count, curr_loss))
            retry_count += 1
        
        # If loss is low enough, return the new model otherwise signal that no new model was found
        if best_loss < max_loss:
            if verbose: self._log_msg('New model found!')
            self._lstm_queue.put((best_model, best_loss))
        else:
            self._lstm_queue.put((None, None))

    def _loop(self, stop_event, verbose, show_progress, fig, window):
                
        # Get past price data for training
        date = datetime.datetime.now().replace(microsecond=0) - datetime.timedelta(days=self._train_days)
        self._btc_actual = self._datastores.btcstock.get_by_range(str(date), verbose=verbose)
        
        # Metadata for the loop
        self._is_running = True
        current_loss = float('inf')
        last_updated = float('-inf')
        iter_counter = 0
        trim_size = self._train_days * 1440
        lstm_lock = threading.Lock()
        data_lock = threading.Lock()
        pred_lock = threading.Lock()
        
        # Start predictions on the next minute
        if verbose: self._log_msg("Waiting until the next minute...")
        time.sleep(60 - datetime.datetime.now().second)
        if verbose: self._log_msg("Starting!")
        
        # Keep running the loop until `stop(...)` is called
        while not self._stop_event.is_set():
            
            # Setup
            if verbose: self._log_msg("Iteration {}".format(iter_counter))
            start_time = datetime.datetime.now()
            
            # Update data for this iteration
            data_lock.acquire()
            self._btc_actual = self._datastores.btcstock.refresh(self._btc_actual, verbose=verbose)
            self._btc_actual = self._btc_actual.tail(trim_size)
            self._is_updated.set()
            data_lock.release()
                
            # Collect times and prices for plotting
            prices = self._btc_actual.open
            ptimes = self._btc_actual.timestamp
    
            # At the start of each iteration, upgrade to a new model if one exists
            if self._lstm_queue.qsize() != 0:
                candidate, new_loss = self._lstm_queue.get()
                self._lstmthread.join()
                if candidate is not None:
                    lstm_lock.acquire()
                    self._lstmoracle = candidate
                    lstm_lock.release()
                    current_loss = new_loss
                    last_updated = time.time()
                else:
                    if verbose: self._log_msg('Retry limit exceeded.')
                    if self._lstmoracle is not None:
                        if verbose: self._log_msg('Staying with current model...')
                        last_updated = time.time()
                    else:
                        if verbose: self._log_msg('No model detected. Starting over...')
    
            # We train / re-train the model if one of the following are true:
            # 1. it is the first iteration
            # 2. it is not already being trained and enough time has passed
            if self._lstmthread is None or (not self._lstmthread.is_alive() and (time.time() - last_updated) / 60 >= self._rfrsh_rate):
                self._lstmthread = threading.Thread(
                    target=self._retrain,
                    args=(prices.to_numpy(), show_progress, verbose, )
                )
                self._lstmthread.start()

            # Make a prediction and update figure data if we have a model. 
            # Otherwise, just use the current price as our prediction for 
            # the next time step
            pred_lock.acquire()
            if self._lstmoracle is not None:
                features = prices.to_numpy()[-self._look_backp:].reshape((1, self._look_backp, 1))
                self._projection.append(self._lstmoracle.predict(features)[0][0])
            else:
                self._projection.append(prices.iloc[-1])
            self._proj_times.append(str(ptimes.iloc[-1] + datetime.timedelta(minutes=1)))
            self._has_predic.set()
            pred_lock.release()

            # Update the figure if one exists
            if fig is not None and window > 0:
                fig.data[0].y = prices[-window:]
                fig.data[0].x = ptimes[-window:]
                fig.data[1].y = self._projection
                fig.data[1].x = self._proj_times
    
            # Display statistics
            if verbose:
                self._log_msg("Total time for iteration: {}s".format(datetime.datetime.now().second - start_time.second))
                self._log_msg("Model loss: {}".format(current_loss))
                
            # If this iteration took too long, exit
            diff = (datetime.datetime.now() - start_time).seconds
            if diff >= 60:
                self._log_msg("Iteration took longer than a minute! Exiting...")
                self._stop_event.set()
                
            # Wait until the next minute to do something
            iter_counter += 1
            time.sleep(55 - datetime.datetime.now().second)
            self._has_predic.clear()
            self._is_updated.clear()
            self._history_df = None
            time.sleep(60 - datetime.datetime.now().second)
            
        # Wait for the training thread to finish
        self._lstmthread.join()
        self._is_running = False

    def accuracy(self, step, offset=0):
        """
        Computes the mean directional accuracy of the current model over the past 
        `days_for_training` * 1440 minutes. The mean directional accuracy is defined 
        as the proportion of times the model correctly predicted that the price of 
        Bitcoin will rise, drop, or stay the same.

        Parameter(s):
        -------------
            step : int (must be greater than 0)
                If specified, computes the accuracy using every `step` minutes.

            offset : int
                If specified, computes the accuracy using the past `offset` minutes. 

        Returns:
        --------
            If the oracle is running and a model exists, returns a floating point 
            value representing the mean directional accuracy of the model. This function
            will wait until computations for the current minute are ready before 
            returning. If no model is ready or this function is called before `run(...)`
            or after `stop(...)`, then None is returned.

        Notes:
        ------
            If the step size is too low, then the accuracy score will drop significantly due 
            to noise. With that in mind, it is recommended to use slightly higher step 
            sizes.
        """
        if self._is_running and not self._stop_event.is_set():
            history_data = self.history()
            if history_data is not None:
                history_data = history_data.iloc[-offset::step]
                true_changes = history_data['actual'].diff().shift(-1)
                pred_changes = history_data['prediction'].shift(-1) - history_data['actual']
                return np.mean((true_changes < 0) == (pred_changes < 0))

    def accuracy_since_deployment(self, step, offset=0):
        """
        Computes the mean directional accuracy using all predictions made since `run(...)` was 
        called. The mean directional accuracy is defined as the proportion of times the model
        correctly predicted that the price of Bitcoin will rise, drop, or stay the same.

        Parameter(s):
        -------------
            step : int (must be greater than 0)
                If specified, computes the accuracy using every `step` minutes.

            offset : int
                If specified, computes the accuracy using the past `offset` minutes. 

        Returns:
        --------
            If the oracle is running, returns a floating point value representing the mean 
            directional accuracy of the model. This function will wait until computations for 
            the current minute are ready before returning. If no model is ready or this function 
            is called before `run(...)` or after `stop(...)`, then None is returned.

        Notes:
        ------
            If the step size is too low, then the accuracy score will drop significantly due 
            to noise. With that in mind, it is recommended to use slightly higher step sizes.
        """
        if self._is_running and not self._stop_event.is_set():
            self._has_predic.wait()
            predict_data = self.predictions_so_far()
            if predict_data is not None:
                predict_data = predict_data.iloc[:-1]
                if len(predict_data) > 1:
                    predict_data = predict_data.iloc[-offset::step]
                    true_changes = predict_data['actual'].diff().shift(-1)
                    pred_changes = predict_data['prediction'].shift(-1) - predict_data['actual']
                    return np.mean((true_changes < 0) == (pred_changes < 0))

    def smape(self, step=1, offset=0):
        """
        Computes the symmetric mean absolute percentage error (SMAPE or sMAPE) for
        the current model over the past `days_for_training` * 1440 minutes.

        Parameter(s):
        -------------
            step : int (must be greater than 0)
                If specified, computes the SMAPE using every `step` minutes.

            offset : int
                If specified, computes the SMAPE using the past `offset` minutes. 

        Returns:
        --------
            If the oracle is running and a model exists, returns a floating point 
            value bounded between 0 and 100 representing the SMAPE of the current
            model. This function will wait until computations for the current minute 
            are ready before returning. If no model is ready or this function is
            called before `run(...)` or after `stop(...)`, then None is returned.
        """
        if self._is_running and not self._stop_event.is_set():
            hstry = self.history()
            if hstry is not None:
                hstry = hstry.iloc[-offset::step]
                numer = np.abs(hstry['prediction'] - hstry['actual'])
                denom = np.abs(hstry['actual']) + np.abs(hstry['prediction'])
                return 100 * np.mean( numer / denom )

    def smape_since_deployment(self, step=1, offset=0):
        """
        Computes the symmetric mean absolute percentage error (SMAPE or sMAPE) 
        using all predictions made since `run(...)` was called.

        Parameter(s):
        -------------
            step : int (must be greater than 0)
                If specified, computes the SMAPE using every `step` minutes.

            offset : int
                If specified, computes the SMAPE using the past `offset` minutes. 

        Returns:
        --------
            If the oracle is running and a model exists, returns a floating point 
            value bounded between 0 and 100 representing the SMAPE of the current
            model. This function will wait until computations for the current minute 
            are ready before returning. If no model is ready or this function is
            called before `run(...)` or after `stop(...)`, then None is returned.
        """
        if self._is_running and not self._stop_event.is_set():
            preds = self.predictions_so_far()
            if preds is not None:
                preds = preds.iloc[:-1]
                if len(preds) > 0:
                    preds = preds.iloc[-offset::step]
                    numer = np.abs(preds['prediction'] - preds['actual'])
                    denom = np.abs(preds['actual']) + np.abs(preds['prediction'])
                    return 100 * np.mean( numer / denom )

    def noise(self, step=1, offset=0):
        """
        Computes the noise of the current model over the past `days_for_training` * 1440
        minutes. Let p[t] denote the prediction of the model at time t > 0. The noise of 
        a model is computed using the following formula:

            mean( abs(p[2] - p[1]), abs(p[3] - p[2]), ..., abs(p[n] - p[n - 1]) )

        If a model has a high amount of noise, then its past predictions have been more
        spread out minute by minute. If a model has a low amount of noise its past 
        predictions have been less spread out minute by minute.

        Parameter(s):
        -------------
            step : int (must be greater than 0)
                If specified, computes the noise using every `step` minutes.

            offset : int
                If specified, computes the noise using the past `offset` minutes.

        Returns:
        --------
            If the oracle is running and a model exists, returns a floating point value 
            representing the noise of the model's predictions. This function will wait 
            until computations for the current minute are ready before returning. If no
            model is ready or this function is called before `run(...)` or after 
            `stop(...)`, then None is returned.
        """
        if self._is_running and not self._stop_event.is_set():
            hstry = self.history()
            if hstry is not None:
                hstry = hstry.iloc[-offset::step]
                return hstry['prediction'].diff().abs().mean()

    def noise_since_deployment(self, step=1, offset=0):
        """
        Computes the noise of the current model using all predictions made since `run(...)`
        was called. Let p[t] denote the prediction of the model at time t > 0. The noise 
        of a model is computed using the following formula:

            mean( abs(p[2] - p[1]), abs(p[3] - p[2]), ..., abs(p[n] - p[n - 1]) )

        If a model has a high amount of noise, then its past predictions have been more
        spread out minute by minute. If a model has a low amount of noise its past 
        predictions have been less spread out minute by minute.

        Parameter(s):
        -------------
            step : int (must be greater than 0)
                If specified, computes the noise using every `step` minutes.

            offset : int
                If specified, computes the noise using the past `offset` minutes.

        Returns:
        --------
            If the oracle is running and a model exists, returns a floating point value 
            representing the noise of the model's predictions. This function will wait 
            until computations for the current minute are ready before returning. If no
            model is ready or this function is called before `run(...)` or after 
            `stop(...)`, then None is returned.
        """
        if self._is_running and not self._stop_event.is_set():
            preds = self.predictions_so_far()
            if preds is not None:
                preds = preds.iloc[:-1]
                if len(preds) > 0:
                    preds = preds.iloc[-offset::step]
                    return preds['prediction'].diff().abs().mean()

    def features(self):
        """
        Get the feature matrix currently being used by the oracle as 
        a labeled pandas dataframe. An extra column of dates is added
        for reference.
        
        Returns:
        --------
            If the oracle is running, returns a pandas dataframe containing the
            feature matrix data. If the dataframe is not ready yet, this function
            will wait until it is ready. If this function is called before `run(...)`
            or after `stop(...)`, then None is returned.
        """
        if self._is_running and not self._stop_event.is_set():
            self._is_updated.wait()
            return self._btc_actual.tail(self._train_days * 1440)\
                       .loc[:, ['timestamp', 'open']]\
                       .reset_index(drop=True)
    
    def predict(self):
        """
        Gets the oracle's most recent prediction for Bitcoin's price.
        
        Returns:
        --------
            If the oracle is running, returns a floating point value representing
            the model's latest prediction of the price of Bitcoin in the next minute.
            If the prediction is not ready yet, this function will wait until it is 
            ready. If this function is called before `run(...)` or after `stop(...)`,
            then None is returned.
        """
        if self._is_running and not self._stop_event.is_set():
            self._has_predic.wait()
            return self._projection[-1]

    def predict_pct(self):
        """
        Gets the oracle's most recent prediction for the percent change in Bitcoin's 
        price.
        
        Returns:
        --------
            If the oracle is running, returns a floating point value representing
            the model's projected percent change for the price of Bitcoin in the 
            next minute. If the prediction is not ready yet, this function will wait
            until it is ready. If this function is called before `run(...)` or after
            `stop(...)`, then None is returned.
        """
        if self._is_running and not self._stop_event.is_set():
            self._has_predic.wait()
            last = self._btc_actual.tail(1)['open'].values[0]
            return 100 * ((self._projection[-1] - last) / abs(last))

    def predictions_so_far(self):
        """
        Gets a dataframe of all predictions made by the oracle since `run(...)` was
        called.

        Returns:
        --------
            If the oracle is running, returns a pandas dataframe with the following 
            columns: 'time', 'prediction', 'actual'. If no model is ready or this function
            is called before `run(...)` or after `stop(...)`, then None is returned.
        """
        if self._is_running and not self._stop_event.is_set():
            self._has_predic.wait()
            start_time = datetime.datetime.strptime(self._proj_times[0], "%Y-%m-%d %H:%M:%S")
            price_vals = self._btc_actual.loc[self._btc_actual['timestamp'] >= start_time, 'open']
            return pd.DataFrame({
                'time' : self._proj_times,
                'prediction' : self._projection,
                'actual' : list(price_vals) + [None]
            })
    
    def history(self, window=None):
        """
        Using the current model, get a dataframe of all predictions it would
        have made in the past `window` minutes. If `window` is None (the 
        default), the model will use the past `days_to_train` * 1440 minutes.
        
        Parameter(s):
        -------------
            window : int or None
                The number of recent data points to use. If None, use the
                entire feature matrix.
        
        Returns:
        --------
            If the oracle is running and a model exists, returns a pandas dataframe 
            with the following columns: 'time', 'prediction', 'actual'. If no
            model is ready or this function is called before `run(...)` or after 
            `stop(...)`, then None is returned.
            
        Notes:
        ------
            If `window` is less than the number of look back points, this
            function will raise an error.
        """
        if self._lstmoracle is not None and not self._stop_event.is_set():
            self._is_updated.wait()
            if self._history_df is None:
                oracle = self._lstmoracle
                actual = self._btc_actual
                prices = actual.open.to_numpy()[-(self._train_days * 1440):]
                ptimes = actual.timestamp.to_numpy()[-(self._train_days * 1440):]
                histry = oracle.predict(
                    TimeseriesGenerator(
                        prices.reshape(-1, 1),
                        prices.reshape(-1, 1),
                        length=self._look_backp,
                        batch_size=self._batch_size
                    )
                )
                self._history_df = pd.DataFrame({
                    'time'       : ptimes[self._look_backp:],
                    'actual'     : prices[self._look_backp:],
                    'prediction' : histry.reshape((-1))
                })  
            return self._history_df.copy() if window is None else self._history_df.tail(window).copy()
    
    def has_model(self):
        """
        Determines if this oracle has a fully trained model.
        
        Returns:
        --------
            True if a model is available and False otherwise.
        """
        if self._is_running and not self._stop_event.is_set():
            return self._lstmoracle is not None
    
    def run(self, verbose=False, show_progress=0, fig=None, window=1440):
        """
        Runs the oracle in the background. This will clear any past
        progress from previous runs.
        
        Parameter(s):
        -------------
            verbose : bool
                If True, displays various metadata such as time per iteration,
                model loss, etc.
            
            show_progress : int
                Either 0, 1, or 2. Passed as the verbose option in keras' `fit(...)`
                and `evaluate(...)` functions. If specified, shows model training and
                model evaluation progress.
            
            fig : plotly.graph_object.FigureWidget
                If specified, plots predictions in realtime. See note below.
                
            window : int
                Controls the number of data points displayed on the figure. 
                Default is one full day (1440 minutes).
                
        Returns:
        --------
            None. Gives control back to the caller immediately.
            
        Notes:
        ------
            To display predictions in realtime, run the code below
            in its own cell.
                
                oracle = MinutelyOracle()
                fig = go.FigureWidget(layout=go.Layout(
                    title=go.layout.Title(text="Live BTC Trading Prices")
                ))
                fig.add_scatter(name='actual')
                fig.add_scatter(name='prediction')
                oracle.run(verbose=True, fig=fig)
                fig
        """
        if not self._is_running:
            # Turns off annoying tensorflow warning messages:
            # https://stackoverflow.com/questions/48608776/how-to-suppress-tensorflow-warning-displayed-in-result
            if not show_progress: tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            self._projection = []
            self._proj_times = []
            self._lstmoracle = None
            self._lstmthread = None
            self._btc_actual = None
            self._featuremtx = None
            self._lstm_queue = queue.Queue()
            self._stop_event.clear()
            self._is_updated.clear()
            self._has_predic.clear()
            self._loopthread = threading.Thread(
                target=self._loop,
                args=(self._stop_event, verbose, show_progress, fig, window, )
            )
            self._loopthread.start()
    
    def stop(self, wait=True):
        """
        Stops the model as soon as possible. By default, this function waits
        for the current iteration and any in-progress model training to complete
        before returning. If this is not desired, set `wait` equal to False to 
        return once all work for the current iteration is complete.
        
        Parameter(s):
        -------------
            wait : bool
                If True, wait for any in-progress training to complete. If
                False, only wait for the current iteration to complete.
                
        Returns:
        --------
            None.
        """
        if self._is_running and not self._stop_event.is_set():
            self._has_predic.wait()
            self._stop_event.set()
            if self._loopthread is not None and wait:
                self._loopthread.join()
