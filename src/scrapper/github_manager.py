import sys
import os
SRC_DIR_GH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR_GH = os.path.dirname(SRC_DIR_GH)
if BASE_DIR_GH not in sys.path: sys.path.append(BASE_DIR_GH)
# Agora importa config
from config import GITHUB_TOKEN, GITHUB_REPO_NAME # Deve pegar do ambiente ou .env
from github import Github, UnknownObjectException, GithubException, RateLimitExceededException
import pandas as pd
from io import StringIO
from typing import Optional
from dotenv import load_dotenv

# Tenta carregar .env ANTES de importar config que usa as variáveis
# (Se não existir, ignora o erro)
try:
     dotenv_path = os.path.join(BASE_DIR_GH, '.env')
     if os.path.exists(dotenv_path):
          print(f"[GitHubManager] Carregando variáveis de: {dotenv_path}")
          load_dotenv(dotenv_path=dotenv_path, override=True) # override=True pode ser útil
     else:
          print(f"[GitHubManager] Arquivo .env não encontrado em {dotenv_path}. Usando variáveis de ambiente existentes.")
except ImportError:
     print("[GitHubManager] Biblioteca python-dotenv não instalada. Usando variáveis de ambiente existentes.")

class GitHubManager:
    def __init__(self):
        """Inicializa a conexão com o GitHub."""

        # Pega as variáveis que DEVERIAM ter sido carregadas do ambiente/dotenv
        self.token = GITHUB_TOKEN
        self.repo_name = GITHUB_REPO_NAME
        self.github: Optional[Github] = None
        self.repo = None

        if not self.token or not self.repo_name:
             # Lança erro se a configuração estiver incompleta APÓS tentar carregar
             raise ValueError("GITHUB_TOKEN ou GITHUB_REPO_NAME não estão configurados no ambiente ou .env.")

        try:
            self.github = Github(self.token, timeout=30)
            user = self.github.get_user()

            if '/' not in self.repo_name:
                 raise ValueError(f"Formato inválido para GITHUB_REPO_NAME: '{self.repo_name}'. Use 'usuario/nome_repo'.")
            self.repo = self.github.get_repo(self.repo_name)
            print(f"✅ Conectado ao GitHub como '{user.login}' e ao repositório: '{self.repo.full_name}'")
        
        except RateLimitExceededException:
             print("❌ Erro: Limite de taxa da API do GitHub excedido. Tente novamente mais tarde.")
             raise ConnectionError("Limite de taxa da API do GitHub excedido.") from None
        
        except GithubException as e:
            status = e.status
            message = e.data.get('message', 'Erro desconhecido')
            print(f"❌ Erro ao conectar ao GitHub ou encontrar repositório: {status} - {message}")
            
            if status == 401: print("   Verifique se o GITHUB_TOKEN está correto e tem as permissões necessárias (repo).")
            
            elif status == 404: print(f"   Verifique se o nome do repositório '{self.repo_name}' está correto e acessível.")
            
            raise ConnectionError(f"Falha na conexão/autenticação com GitHub: {status} - {message}") from e
        
        except Exception as e:
            print(f"❌ Erro inesperado ao inicializar GitHubManager: {str(e)}")
            
            raise ConnectionError(f"Erro inesperado na inicialização do GitHub: {str(e)}") from e

    def save_or_update_github(self, df: pd.DataFrame, path: str, branch: str = "main", commit_message_prefix: str = "") -> bool:
        """Salva (cria ou atualiza) um DataFrame como CSV no GitHub."""
        if not self.repo: return False
        try:
            content_bytes = df.to_csv(index=False).encode('utf-8')
            current_sha = None
            try:
                file_content = self.repo.get_contents(path, ref=branch)
                current_sha = file_content.sha
                if file_content.decoded_content == content_bytes:
                    print(f"ℹ️ Conteúdo de '{path}' no GitHub já está atualizado.")
                    return True
            except UnknownObjectException: pass 
            except GithubException as e_get:
                 print(f"❌ Erro ao verificar arquivo existente '{path}': {e_get.status} {e_get.data.get('message', '')}")
                 return False

            base_filename = os.path.basename(path)
            commit_message = f"{commit_message_prefix}Atualiza {base_filename}" if current_sha else f"{commit_message_prefix}Cria {base_filename}"
            commit_message = commit_message.strip() 

            if current_sha:
                self.repo.update_file(path, commit_message, content_bytes, current_sha, branch=branch)
                print(f"🔄 Dados atualizados no GitHub: {path} (Branch: {branch})")
            else:
                self.repo.create_file(path, commit_message, content_bytes, branch=branch)
                print(f"✅ Novo arquivo criado no GitHub: {path} (Branch: {branch})")
            return True
        except RateLimitExceededException:
             print("❌ Erro: Limite de taxa da API do GitHub excedido ao tentar salvar.")
             return False
        except GithubException as e_save:
            action = "atualizar" if current_sha else "criar"
            print(f"❌ Falha ao {action} '{path}' no GitHub: {e_save.status} {e_save.data.get('message', '')}")
            if e_save.status == 409: print("   Pode ser necessário atualizar branch local ou resolver conflitos.")
            return False
        except Exception as e:
            action = "atualizar" if current_sha else "criar"
            print(f"❌ Erro inesperado ao {action} '{path}': {str(e)}")
            return False

    def read_csv_from_github(self, path: str, branch: str = "main") -> Optional[pd.DataFrame]:
        """Lê um arquivo CSV diretamente do GitHub."""
        
        if not self.repo: return None
        try:
            file_content = self.repo.get_contents(path, ref=branch)
            decoded_content = file_content.decoded_content.decode('utf-8')
            df = pd.read_csv(StringIO(decoded_content))
            print(f"✔️ CSV lido com sucesso do GitHub: '{path}' (Branch: {branch})")
            return df
        except UnknownObjectException:
            print(f"ℹ️ Arquivo não encontrado em '{path}' na branch '{branch}'.")
            return None
        except RateLimitExceededException:
             print(f"❌ Erro: Limite de taxa da API do GitHub excedido ao tentar ler '{path}'.")
             return None
        except GithubException as e:
            print(f"❌ Erro ao ler CSV do GitHub '{path}': {e.status} {e.data.get('message', '')}")
            return None
        except Exception as e:
            print(f"❌ Erro inesperado ao ler CSV do GitHub '{path}': {str(e)}")
            return None