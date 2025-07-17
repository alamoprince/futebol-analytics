from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression 
from sklearn.exceptions import NotFittedError
from typing import Optional, Any

from logger_config import setup_logger
logger = setup_logger("Calibrators")

class BaseCalibrator(ABC):
    """Classe base abstrata para calibradores de probabilidade."""
    def __init__(self):
        self.calibrator_model: Optional[Any] = None

    @abstractmethod
    def fit(self, y_proba_raw_1d: np.ndarray, y_true: np.ndarray) -> 'BaseCalibrator':

        pass

    @abstractmethod
    def predict_proba(self, y_proba_raw_1d: np.ndarray) -> np.ndarray:

        pass

    def is_fitted(self) -> bool:
        return self.calibrator_model is not None

class IsotonicCalibrator(BaseCalibrator):
    def __init__(self):
        super().__init__()
        self.calibrator_model = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        logger.info("IsotonicCalibrator inicializado.")

    def fit(self, y_proba_raw_1d: np.ndarray, y_true: np.ndarray) -> 'IsotonicCalibrator':
        if y_proba_raw_1d is None or y_true is None:
            logger.error("IsotonicCalibrator.fit: y_proba_raw_1d ou y_true é None.")
            raise ValueError("Inputs para fit não podem ser None.")
        if len(y_proba_raw_1d) != len(y_true):
            logger.error("IsotonicCalibrator.fit: Comprimentos de y_proba_raw_1d e y_true não coincidem.")
            raise ValueError("Inputs para fit devem ter o mesmo comprimento.")
        if y_proba_raw_1d.ndim != 1:
            logger.error(f"IsotonicCalibrator.fit: y_proba_raw_1d deve ser 1D, mas tem {y_proba_raw_1d.ndim} dimensões.")
            raise ValueError("y_proba_raw_1d deve ser um array 1D para IsotonicRegression.")

        try:
            self.calibrator_model.fit(y_proba_raw_1d, y_true)
            logger.info("IsotonicCalibrator ajustado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao ajustar IsotonicCalibrator: {e}", exc_info=True)
            self.calibrator_model = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0) # Reseta
        return self

    def predict_proba(self, y_proba_raw_1d: np.ndarray) -> np.ndarray:
        if not self.is_fitted() or not hasattr(self.calibrator_model, 'transform'): 
             logger.error("IsotonicCalibrator.predict_proba: Calibrador não ajustado.")
             raise NotFittedError("IsotonicCalibrator não foi ajustado antes de chamar predict_proba.")
        if y_proba_raw_1d.ndim != 1:
            logger.error(f"IsotonicCalibrator.predict_proba: y_proba_raw_1d deve ser 1D, mas tem {y_proba_raw_1d.ndim} dimensões.")
            raise ValueError("y_proba_raw_1d deve ser um array 1D para IsotonicRegression.")
        
        try:
            calibrated_probas = self.calibrator_model.predict(y_proba_raw_1d)
            return np.clip(calibrated_probas, 0.0, 1.0) 
        except Exception as e:
            logger.error(f"Erro em IsotonicCalibrator.predict_proba: {e}", exc_info=True)
            return y_proba_raw_1d 

class SigmoidCalibrator(BaseCalibrator):

    def __init__(self):
        super().__init__()
        self.calibrator_model = LogisticRegression(solver='liblinear', C=1e6, class_weight=None)
        logger.info("SigmoidCalibrator (Platt) inicializado.")

    def fit(self, y_proba_raw_1d: np.ndarray, y_true: np.ndarray) -> 'SigmoidCalibrator':
        if y_proba_raw_1d is None or y_true is None:
            logger.error("SigmoidCalibrator.fit: y_proba_raw_1d ou y_true é None.")
            raise ValueError("Inputs para fit não podem ser None.")
        if len(y_proba_raw_1d) != len(y_true):
            logger.error("SigmoidCalibrator.fit: Comprimentos não coincidem.")
            raise ValueError("Inputs para fit devem ter o mesmo comprimento.")
        if y_proba_raw_1d.ndim != 1:
            logger.error(f"SigmoidCalibrator.fit: y_proba_raw_1d deve ser 1D, tem {y_proba_raw_1d.ndim}D.")

        X_calib = y_proba_raw_1d.reshape(-1, 1) 
        try:
            self.calibrator_model.fit(X_calib, y_true)
            logger.info("SigmoidCalibrator ajustado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao ajustar SigmoidCalibrator: {e}", exc_info=True)
            self.calibrator_model = LogisticRegression(solver='liblinear', C=1e6, class_weight=None) 
        return self

    def predict_proba(self, y_proba_raw_1d: np.ndarray) -> np.ndarray:
        if not self.is_fitted() or not hasattr(self.calibrator_model, 'classes_'): 
             logger.error("SigmoidCalibrator.predict_proba: Calibrador não ajustado.")
             raise NotFittedError("SigmoidCalibrator não foi ajustado antes de chamar predict_proba.")
        if y_proba_raw_1d.ndim != 1:
            logger.error(f"SigmoidCalibrator.predict_proba: y_proba_raw_1d deve ser 1D, tem {y_proba_raw_1d.ndim}D.")

        X_pred_calib = y_proba_raw_1d.reshape(-1, 1) 
        try:
            idx_positive_class = np.where(self.calibrator_model.classes_ == 1)[0]
            if len(idx_positive_class) == 0:
                logger.error("SigmoidCalibrator: Classe positiva (1) não encontrada nos classes_ do modelo. Usando coluna 1.")
                idx_positive_class = 1
            else:
                idx_positive_class = idx_positive_class[0]

            calibrated_probas_all_classes = self.calibrator_model.predict_proba(X_pred_calib)
            calibrated_probas_positive = calibrated_probas_all_classes[:, idx_positive_class]
            return np.clip(calibrated_probas_positive, 0.0, 1.0)
        except Exception as e:
            logger.error(f"Erro em SigmoidCalibrator.predict_proba: {e}", exc_info=True)
            return y_proba_raw_1d 

def get_calibrator_instance(method_name: str) -> Optional[BaseCalibrator]:

    if method_name is None: 
        return None
    method_name_lower = method_name.lower()
    if method_name_lower == 'isotonic':
        return IsotonicCalibrator()
    elif method_name_lower == 'sigmoid' or method_name_lower == 'platt':
        return SigmoidCalibrator()
    else:
        logger.warning(f"Método de calibração '{method_name}' desconhecido. Retornando None (sem calibração).")
        return None