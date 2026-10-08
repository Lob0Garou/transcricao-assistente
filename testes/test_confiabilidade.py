"""Regressões offline: sem áudio privado, download de modelos ou chamadas de API."""
import json
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ta import env

env.preparar()

from ta import asr, deepseek, juiz  # noqa: E402
import transcrever  # noqa: E402


def alvo():
    return {"id": "a1", "tipo": "palavra", "canal": "M", "ini": 1.0, "fim": 2.0,
            "antes": "hello", "depois": "there", "atual": "world", "variantes": 2,
            "candidatos": [{"texto": "world", "votos": 1}, {"texto": "word", "votos": 1}]}


class JuizTests(unittest.TestCase):
    def test_falha_total_nao_e_sucesso(self):
        with patch.object(juiz, "carregar_chave", return_value="chave-ficticia"), \
             patch.object(juiz, "_post", side_effect=RuntimeError("indisponível")):
            info = juiz.ranquear([alvo()], progresso=lambda _: None)
        self.assertFalse(info["usado"])
        self.assertEqual(info["falhas"], 1)
        self.assertEqual(info["avaliados"], 0)

    def test_resposta_invalida_nao_e_sucesso(self):
        for resp in ({}, {"answers": []}, [], {"answers": {"a1": {"probabilities": []}}}):
            with self.subTest(resp=resp), \
                 patch.object(juiz, "carregar_chave", return_value="chave-ficticia"), \
                 patch.object(juiz, "_post", return_value=resp):
                self.assertFalse(juiz.ranquear([alvo()], progresso=lambda _: None)["usado"])

    def test_probabilidades_validas_sao_aplicadas(self):
        a = alvo()
        resposta = {"answers": {"a1": {"choice": "c1", "probabilities": {"c1": .7, "c2": .3},
                                      "confidence": .8}}}
        with patch.object(juiz, "carregar_chave", return_value="chave-ficticia"), \
             patch.object(juiz, "_post", return_value=resposta):
            info = juiz.ranquear([a], progresso=lambda _: None)
        self.assertTrue(info["usado"])
        self.assertEqual(info["avaliados"], 1)
        self.assertEqual(a["jev"]["escolha"], "world")
        self.assertEqual(a["candidatos"][0]["contexto"], .7)


class ModeloTests(unittest.TestCase):
    def test_gpu_indisponivel_usa_cpu(self):
        chamadas = []
        cpu = object()
        def modelo(nome, **kw):
            chamadas.append(kw)
            if kw["device"] == "cuda":
                raise RuntimeError("GPU indisponível")
            return cpu
        with patch.dict(sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=modelo)}):
            resultado, dispositivo = asr.carregar_modelo("small")
        self.assertIs(resultado, cpu)
        self.assertEqual(dispositivo, "cpu")
        self.assertEqual(chamadas, [{"device": "cuda", "compute_type": "int8_float16"},
                                    {"device": "cpu", "compute_type": "int8"}])


class RevisorTests(unittest.TestCase):
    def test_json_sem_contrato_nao_parece_sucesso(self):
        for texto in ('{}', '{"duvidas": {}}',
                      '{"duvidas": [{"id": [], "alternativas": [1]}]}',
                      '{"duvidas": [], "outros_erros": [1]}'):
            with self.subTest(texto=texto), self.assertRaises(ValueError):
                deepseek._extrair_json(texto)

    def test_revisao_vazia_valida_e_aceita(self):
        self.assertEqual(deepseek._extrair_json('{"duvidas": []}'), {"duvidas": []})

    def test_provedor_com_choices_vazio_nao_quebra_pipeline(self):
        with patch.object(deepseek.urllib.request, "urlopen") as abrir:
            abrir.return_value.__enter__.return_value.read.return_value = b'{"choices": []}'
            dados, motivo = deepseek._post({"base": "https://exemplo.test/v1"},
                                           "chave-ficticia", {}, 1)
        self.assertIsNone(dados)
        self.assertIn("formato inválido", motivo)


class CacheTests(unittest.TestCase):
    def test_modo_local_nao_herda_juiz_do_cache_externo(self):
        with tempfile.TemporaryDirectory() as pasta:
            antigo = alvo()
            antigo["jev"] = {"escolha": "word", "confianca": .99}
            antigo["candidatos"][0]["contexto"] = .01
            for nome in ("cand_small_auto.json", "cand_small_auto_ds1_jev1.json"):
                with open(os.path.join(pasta, nome), "w", encoding="utf-8") as f:
                    json.dump({"palavra|M|1.00|2.00": antigo}, f)
            a = SimpleNamespace(modelo="small", idioma="auto", refazer=False,
                                sem_deepseek=True, sem_jev=True)
            with patch.object(transcrever.candidatos, "alvos_da_analise", return_value=[alvo()]), \
                 patch.object(transcrever.candidatos, "gerar"), \
                 patch.object(transcrever.candidatos, "pontuar_candidatos"), \
                 patch.object(transcrever.deepseek, "revisar") as ds, \
                 patch.object(transcrever.juiz, "ranquear") as jev:
                alvos, info = transcrever.resolver_duvidas(a, None, {"falas": []}, pasta, None,
                                                         {"M": "en"}, "en")
            ds.assert_not_called()
            jev.assert_not_called()
            self.assertNotIn("jev", alvos[0])
            self.assertNotIn("contexto", alvos[0]["candidatos"][0])
            self.assertFalse(info["jev"]["usado"])
            self.assertTrue(os.path.isfile(os.path.join(pasta, "cand_small_auto_ds0_jev0.json")))


if __name__ == "__main__":
    unittest.main()
