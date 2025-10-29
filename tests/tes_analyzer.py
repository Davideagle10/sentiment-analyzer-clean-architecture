import openai
import os
import json
from abc import ABC, abstractmethod
from typing import Union, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from colorama import Fore, Style, init
import logging

# Importar el nuevo ConfigManager
from config.config_manager import ConfigManager

# Inicializar colorama para colores cross-platform
init(autoreset=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# ENUMS Y DATA CLASSES 
class SentimentLevel(Enum):
    VERY_NEGATIVE = "Muy Negativo"
    NEGATIVE = "Negativo"
    SLIGHTLY_NEGATIVE = "Ligeramente Negativo"
    NEUTRAL = "Neutral"
    SLIGHTLY_POSITIVE = "Ligeramente Positivo"
    POSITIVE = "Positivo"
    VERY_POSITIVE = "Muy Positivo"
    ERROR = "Error"

@dataclass
class SentimentResult:
    level: SentimentLevel
    score: float
    color: str
    description: str

@dataclass
class AnalysisRequest:
    text: str
    max_tokens: int = 8
    temperature: float = 0.1

@dataclass
class AnalysisResponse:
    success: bool
    score: Optional[float] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None

#  PRINCIPIO DE RESPONSABILIDAD ÚNICA (SRP) 
class OpenAIClient:
    """Responsable única de comunicarse con OpenAI"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.client = openai.OpenAI(api_key=config.settings.openai_api_key)
        self.conversation_history: List[Dict] = [
            {"role": "system", "content": config.settings.openai_system_prompt}
        ]
    
    def analyze_sentiment(self, request: AnalysisRequest) -> AnalysisResponse:
        """Analiza el sentimiento del texto usando OpenAI"""
        try:
            logger.info(f"Analizando sentimiento para texto: {request.text[:50]}...")
            
            self.conversation_history.append({
                "role": "user", 
                "content": request.text
            })
            
            response = self.client.chat.completions.create(
                model=self.config.settings.openai_model,
                messages=self.conversation_history,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                timeout=self.config.settings.request_timeout
            )
            
            answer = response.choices[0].message.content.strip()
            logger.debug(f"Respuesta cruda de OpenAI: {answer}")
            
            self.conversation_history.append({
                "role": "assistant", 
                "content": answer
            })
            
            # Parsear la respuesta
            score = self._parse_sentiment_score(answer)
            
            return AnalysisResponse(
                success=True,
                score=score,
                raw_response=answer
            )
            
        except openai.APIConnectionError as e:
            error_msg = f"Error de conexión: {e}"
            logger.error(error_msg)
            return AnalysisResponse(success=False, error=error_msg)
        except openai.RateLimitError as e:
            error_msg = f"Límite de tasa excedido: {e}"
            logger.error(error_msg)
            return AnalysisResponse(success=False, error=error_msg)
        except openai.APIError as e:
            error_msg = f"Error de API: {e}"
            logger.error(error_msg)
            return AnalysisResponse(success=False, error=error_msg)
        except Exception as e:
            error_msg = f"Error inesperado: {e}"
            logger.error(error_msg)
            return AnalysisResponse(success=False, error=error_msg)
    
    def _parse_sentiment_score(self, answer: str) -> float:
        """Parsea la respuesta para extraer el score de sentimiento"""
        try:
            # Limpiar la respuesta y convertir a float
            cleaned_answer = ''.join(
                char for char in answer 
                if char.isdigit() or char in ['-', '.', ',']
            ).replace(',', '.')
            
            if not cleaned_answer:
                raise ValueError("Respuesta vacía")
                
            score = float(cleaned_answer)
            
            # Validar rango
            if not -1 <= score <= 1:
                logger.warning(f"Score fuera de rango: {score}")
                score = max(-1.0, min(1.0, score))  # Clamp al rango
                
            return score
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error parseando score: {e}, respuesta: {answer}")
            raise ValueError(f"No se pudo parsear el score de la respuesta: {answer}")

# PRINCIPIO ABIERTO/CERRADO (OCP) 
class SentimentAnalyzer(ABC):
    """Interface para diferentes estrategias de análisis"""
    
    @abstractmethod
    def analyze(self, score: float) -> SentimentResult:
        pass

class GradientSentimentAnalyzer(SentimentAnalyzer):
    """Implementación con análisis por gradientes"""
    
    def analyze(self, score: float) -> SentimentResult:
        if score < -0.8:
            return SentimentResult(
                level=SentimentLevel.VERY_NEGATIVE,
                score=score,
                color=Fore.RED,
                description="Sentimiento muy negativo"
            )
        elif score < -0.3:
            return SentimentResult(
                level=SentimentLevel.NEGATIVE,
                score=score,
                color=Fore.RED,
                description="Sentimiento negativo"
            )
        elif score < -0.1:
            return SentimentResult(
                level=SentimentLevel.SLIGHTLY_NEGATIVE,
                score=score,
                color=Fore.YELLOW,
                description="Sentimiento ligeramente negativo"
            )
        elif score <= 0.1:
            return SentimentResult(
                level=SentimentLevel.NEUTRAL,
                score=score,
                color=Fore.WHITE,
                description="Sentimiento neutral"
            )
        elif score <= 0.4:
            return SentimentResult(
                level=SentimentLevel.SLIGHTLY_POSITIVE,
                score=score,
                color=Fore.GREEN,
                description="Sentimiento ligeramente positivo"
            )
        elif score <= 0.9:
            return SentimentResult(
                level=SentimentLevel.POSITIVE,
                score=score,
                color=Fore.GREEN,
                description="Sentimiento positivo"
            )
        else:
            return SentimentResult(
                level=SentimentLevel.VERY_POSITIVE,
                score=score,
                color=Fore.CYAN,
                description="Sentimiento muy positivo"
            )

# PRINCIPIO DE SEGREGACIÓN DE INTERFACES (ISP) 
class DisplayInterface(ABC):
    """Interface específica para display"""
    
    @abstractmethod
    def show_input_prompt(self):
        pass
    
    @abstractmethod
    def show_result(self, result: SentimentResult):
        pass
    
    @abstractmethod
    def show_error(self, error: str):
        pass
    
    @abstractmethod
    def show_welcome(self):
        pass

class ConsoleDisplay(DisplayInterface):
    """Implementación para consola con colores"""
    
    def show_input_prompt(self):
        print(f"\n{Fore.YELLOW}Dime algo (o 'salir' para terminar):{Style.RESET_ALL} ", end="")
    
    def show_result(self, result: SentimentResult):
        print(f"{Fore.WHITE}Sentimiento: {result.color}{result.level.value} "
              f"{Fore.WHITE}({result.score:.2f}) - {result.description}")
    
    def show_error(self, error: str):
        print(f"{Fore.RED}❌ Error: {error}")
    
    def show_welcome(self):
        print(f"\n{Fore.CYAN}{'='*50}")
        print(f"{Fore.CYAN}          ANALIZADOR DE SENTIMIENTOS")
        print(f"{Fore.CYAN}{'='*50}")
        print(f"{Fore.WHITE}Usando OpenAI para analizar sentimientos")
        print(f"{Fore.WHITE}Escribe cualquier texto y analizaré su sentimiento")
        print(f"{Fore.YELLOW}El score va de -1 (muy negativo) a 1 (muy positivo)")
        print(f"{Fore.CYAN}{'='*50}")

# PRINCIPIO DE INVERSIÓN DE DEPENDENCIAS (DIP) 
class SentimentAnalysisService:
    """Servicio principal que coordina el análisis"""
    
    def __init__(self, 
                 openai_client: OpenAIClient, 
                 analyzer: SentimentAnalyzer):
        self.openai_client = openai_client
        self.analyzer = analyzer
    
    def analyze_text(self, text: str) -> tuple[Optional[SentimentResult], Optional[str]]:
        """Analiza el texto y retorna el resultado o error"""
        request = AnalysisRequest(text=text)
        response = self.openai_client.analyze_sentiment(request)
        
        if response.success and response.score is not None:
            result = self.analyzer.analyze(response.score)
            return result, None
        else:
            return None, response.error

class SentimentAnalysisApp:
    """Clase principal de la aplicación"""
    
    def __init__(self, 
                 service: SentimentAnalysisService,
                 display: DisplayInterface):
        self.service = service
        self.display = display
        self.running = False
    
    def run(self):
        """Ejecuta la aplicación principal"""
        self.running = True
        self.display.show_welcome()
        
        while self.running:
            try:
                self.display.show_input_prompt()
                user_input = input().strip()
                
                if self._should_exit(user_input):
                    self._shutdown()
                    continue
                
                if not user_input:
                    continue
                
                # Analizar el texto
                result, error = self.service.analyze_text(user_input)
                
                if error:
                    self.display.show_error(error)
                else:
                    self.display.show_result(result)
                    
            except KeyboardInterrupt:
                self._shutdown()
            except Exception as e:
                logger.error(f"Error en loop principal: {e}")
                self.display.show_error(f"Error inesperado: {e}")
    
    def _should_exit(self, user_input: str) -> bool:
        """Determina si la aplicación debe salir"""
        exit_commands = ['salir', 'exit', 'quit', 'q']
        return user_input.lower() in exit_commands
    
    def _shutdown(self):
        """Apaga la aplicación gracefulmente"""
        self.running = False
        print(f"\n{Fore.GREEN}👋 ¡Hasta luego! Gracias por usar el analizador.")

#  FÁBRICA PARA CREAR LA APP 
class AppFactory:
    """Factory para crear la aplicación con las dependencias inyectadas"""
    
    @staticmethod
    def create_app() -> SentimentAnalysisApp:
        config = ConfigManager()
        if not config.validate_config():
            raise ValueError("Configuración inválida. Revisa las variables de entorno.")
        
        # Dependencias
        openai_client = OpenAIClient(config)
        analyzer = GradientSentimentAnalyzer()
        display = ConsoleDisplay()
        service = SentimentAnalysisService(openai_client, analyzer)
        
        return SentimentAnalysisApp(service, display)

# MANEJO DE ERRORES GLOBAL 
def handle_global_exception(exc_type, exc_value, exc_traceback):
    """Manejador global de excepciones"""
    if issubclass(exc_type, KeyboardInterrupt):
        print(f"\n{Fore.YELLOW}Interrumpido por el usuario")
        return
    
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    print(f"{Fore.RED}❌ Error crítico: {exc_value}")

# EJECUCIÓN PRINCIPAL 
if __name__ == "__main__":
    # Configurar manejo global de excepciones
    import sys
    sys.excepthook = handle_global_exception
    
    try:
        app = AppFactory.create_app()
        app.run()
    except Exception as e:
        logger.critical(f"Error crítico al iniciar la aplicación: {e}")
        print(f"{Fore.RED}No se pudo iniciar la aplicación: {e}")
        print(f"{Fore.YELLOW}Asegúrate de tener configurada la variable OPENAI_API_KEY en el archivo .env")
        sys.exit(1)