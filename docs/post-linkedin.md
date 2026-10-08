Uma transcrição pode parecer correta e ainda conter palavras que ninguém disse.

Criei uma ferramenta em Python para me ajudar a analisar áudios e revisar transcrições geradas por IA.

O Whisper processa o áudio localmente. A ferramenta cruza duas passadas de transcrição, marca divergências
e gera alternativas para os trechos duvidosos. DeepSeek revisa o texto, e JEV avalia os candidatos pelo contexto.
Esses dois revisores são opcionais e recebem texto, não o áudio.

Na interface, posso ouvir exatamente aquele trecho, comparar as alternativas e decidir o que entra no texto final.

Durante os experimentos, apareceu um limite interessante: uma alternativa pode fazer mais sentido na frase
e ainda estar errada no áudio. Por isso, a discordância entre áudio e contexto vira um sinal para revisar.

Abri o código com licença MIT, instruções de instalação e testes offline:
https://github.com/Lob0Garou/transcricao-assistente

Ainda é experimental, especialmente na identificação de falantes e na validação em português.
Já tem sido útil para organizar minha revisão e mostrar onde concentrar a escuta.

Se você trabalha com transcrição ou reconhecimento de fala, contribuições e relatos de uso são bem-vindos.
