# VolEngine plano 6a: empacotamento e release (implementação)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Objetivo:** `pip install volengine` funciona nas 7 plataformas, com a mesma velocidade do build local, e uma tag `v*`
publica wheels no PyPI e bibliotecas na GitHub Release. Termina com a `v0.9.0` publicada.

**Arquitetura:**
- Um script (`python/scripts/build_native.py`) compila só o alvo `volengine` e o põe em `python/src/volengine/_lib`.
- Um hook do hatchling dá ao wheel a tag `py3-none-<plataforma>` lida do próprio binário.
- Um workflow reutilizável (`wheels.yml`) monta, testa e guarda os 7 wheels e as 7 bibliotecas. Ele roda em todo push
  para master e é chamado pelo `release.yml`, que publica.

**Stack:** CMake 3.28 e Ninja; containers `manylinux_2_28` e `musllinux_1_2`; MSYS2 ucrt64; Apple Clang; hatchling; uv;
GitHub Actions; Trusted Publishing do PyPI.

**Spec:** `docs/superpowers/specs/2026-11-15-volengine-06-release-design.md` §5.1 (e §1 a §4). Modelo de referência:
`C:/Users/x/Documents/Programs/Csharp/FixedWidthParser`, arquivos `.github/workflows/release.yml`,
`.github/workflows/native-bindings.yml`, `python/hatch_build.py` e `python/scripts/check_wheel_tag.py`.

## Restrições globais

- Plataformas e tags:

  | Plataforma | Tag do wheel |
  |---|---|
  | Windows x64 | `win_amd64` |
  | macOS arm64 | `macosx_11_0_arm64` |
  | macOS x64 | `macosx_11_0_x86_64` |
  | Linux glibc x64 | `manylinux_2_28_x86_64` |
  | Linux glibc arm64 | `manylinux_2_28_aarch64` |
  | Linux musl x64 | `musllinux_1_2_x86_64` |
  | Linux musl arm64 | `musllinux_1_2_aarch64` |

- Fallback: se o gcc-toolset-14 do `manylinux_2_28` não compilar, usar `manylinux_2_34`. A troca e o motivo ficam
  registrados no commit e no §10 do `benchmarks.md`.
- `MACOSX_DEPLOYMENT_TARGET=11.0`. Só sobe por erro de availability do libc++, com o motivo registrado.
- Nada de `-march`/`-mcpu`/`-mavx*` em flag nenhuma. O Highway despacha em runtime.
- Sem sdist.
- Python: `requires-python = ">=3.11"`, CI em 3.14 e 3.11, release testa o wheel em 3.14.
- Nomes dos assets: `volengine-native-<os>-<arch>.<ext>`.
  - `os` é `win`, `osx`, `linux` ou `linux-musl`.
  - `arch` é `x64` ou `arm64`.
  - No Windows vão também `volengine-native-win-x64.dll.a` e `volengine-native-win-x64.lib`. A release tem então
    9 assets mais o SBOM.
- O `.def` do Windows sai do linker (`-Wl,--output-def`) e só serve para gerar o `.lib`; não vira asset. Isso desvia
  da spec, que gera o `.def` a partir do contrato da ABI: o contrato só existe no 6c, que passa a gerar ou conferir o
  `.def` a partir dele.
- O nome do asset, o SONAME e o `install_name` concordarem com o que o loader procura é trabalho do 6b (o `cpp/` é
  quem baixa o asset). O 6a só renomeia.
- Tag com `-` vira prerelease na GitHub Release. A versão PEP 440 do wheel é `str(packaging.version.Version(tag sem
  "v"))`, por exemplo `v1.0.0-rc.1` vira `1.0.0rc1`.
- Versão no repositório: `python/pyproject.toml` em `0.9.0.dev0`; CMake com `VE_VERSION` em `0.9.0`.
- Interface C intocada: nenhuma assinatura nova neste plano.
- Push para master e criação de tag só com o ok explícito do autor, pedido no passo que faz.
- Commits sem trailer `Co-Authored-By` (regra do autor).

## Foco da revisão

1. **Wheel testado contra o código-fonte, e não contra o wheel** (falso verde): o pytest do job tem de importar o
   `volengine` de `site-packages` e carregar a biblioteca de `site-packages/volengine/_lib`. Teste na Task 3, passo 6,
   e no workflow da Task 4.
2. **Tag que mente sobre o sistema mínimo:** a `.so` exige `GLIBC_2.N` com N > 28, ou a `.dylib` declara macOS acima de
   11.0. O job falha comparando a tag do wheel com a esperada. Testes das funções de piso na Task 3; checagem no
   workflow na Task 4.
3. **DLL que só carrega numa máquina com MSYS2:** importa `libstdc++-6.dll`, `libgcc_s_seh-1.dll` ou
   `libwinpthread-1.dll`; o mesmo vale para `libstdc++.so.6` e `libgcc_s.so.1` no Linux. `check_native_deps.py` falha
   (Task 1) e roda em toda plataforma (Task 4).
4. **Versão de prerelease que o PyPI rejeita ou normaliza diferente da tag:** `v1.0.0-rc.1` tem de virar `1.0.0rc1`
   no wheel e na mensagem do job. Teste em `test_release_version.py` (Task 5).
5. **Binário com instrução fora do baseline** (crash numa CPU sem AVX2): nenhuma flag `-march`. `build_native.py`
   recusa uma flag dessas vinda do ambiente (`CFLAGS`/`CXXFLAGS`) (Task 1).

---

### Task 1: build nativo do wheel, runtime estático e checagem de dependências

**Arquivos:**
- Modificar: `CMakeLists.txt` (versão, runtime estático fora do Windows, `.def` no MinGW)
- Criar: `python/scripts/build_native.py`
- Criar: `python/scripts/check_native_deps.py`
- Testar: `python/tests/test_packaging_scripts.py`

**Interfaces:**
- Produz:
  - `build_native.py [--build-dir DIR]`: configura com `-G Ninja -DCMAKE_BUILD_TYPE=Release -DVE_BUILD_TESTS=OFF`,
    compila `--target volengine` e copia a biblioteca para `python/src/volengine/_lib/`. Saída padrão:
    `build/wheel-native`. No MinGW copia também `libvolengine.dll.a` e `volengine.def`.
  - `check_native_deps.py PATH`: termina com 0 ou 1. Imprime `ok: <deps>` ou `error: <dep proibida>`.
  - A função `forbidden(system: str, needed: list[str]) -> list[str]`, testável sem binário.

- [ ] **Passo 1: Escrever os testes que falham** em `python/tests/test_packaging_scripts.py`. Eles importam os scripts
  por caminho, com `importlib.util.spec_from_file_location`.

```python
def test_windows_mingw_runtime_is_forbidden():
    assert deps.forbidden("Windows", ["KERNEL32.dll", "libstdc++-6.dll", "libwinpthread-1.dll"]) == [
        "libstdc++-6.dll", "libwinpthread-1.dll"]

def test_windows_system_dlls_pass():
    assert deps.forbidden("Windows", ["KERNEL32.dll", "msvcrt.dll", "api-ms-win-crt-runtime-l1-1-0.dll"]) == []

def test_linux_shared_libstdcxx_is_forbidden():
    assert deps.forbidden("Linux", ["libstdc++.so.6", "libgcc_s.so.1", "libm.so.6", "libc.so.6"]) == [
        "libstdc++.so.6", "libgcc_s.so.1"]

def test_macos_allows_only_system_paths():
    assert deps.forbidden("Darwin", ["/usr/lib/libc++.1.dylib", "/opt/homebrew/lib/libomp.dylib"]) == [
        "/opt/homebrew/lib/libomp.dylib"]

def test_build_refuses_isa_flags():
    with pytest.raises(SystemExit, match="-march"):
        build.check_flags({"CXXFLAGS": "-O2 -march=native"})
```

- [ ] **Passo 2: Rodar e ver falhar**

  Rodar: `uv run --project python pytest python/tests/test_packaging_scripts.py -q`. Se o uv ainda não estiver no
  projeto (Task 2), usar o pytest atual. Esperado: FAIL, porque os módulos não existem.

- [ ] **Passo 3: Implementar.**
  - **`check_native_deps.py`:**
    - Lê as dependências com `objdump -p` (Windows/MinGW: linhas `DLL Name:`), `readelf -d` (Linux: `NEEDED`) ou
      `otool -L` (macOS, pulando a primeira linha, que é o próprio id).
    - Proíbe: no Windows, nomes que começam com `libstdc++`, `libgcc` ou `libwinpthread`; no Linux, `libstdc++` e
      `libgcc_s`; no macOS, caminhos fora de `/usr/lib/` e `/System/`.
  - **`build_native.py`:**
    - `check_flags(environ: Mapping[str, str]) -> None` sai com mensagem se `CFLAGS` ou `CXXFLAGS` contiverem
      `-march`, `-mcpu`, `-mavx` ou `-mtune`.
    - Depois chama `cmake`; no Windows usa `gcc`/`g++` do PATH (ucrt64).
  - **`CMakeLists.txt`:**
    - `set(VE_VERSION 0.9.0 CACHE STRING ...)` e `project(VolEngine VERSION ${VE_VERSION} ...)`. O `project()` tem de
      ler a cache, então o `set` vem antes dele.
    - Para `NOT WIN32 AND NOT APPLE`, `target_link_options(volengine PRIVATE -static-libstdc++ -static-libgcc)`.
    - No bloco `if(MINGW)`, acrescentar `-Wl,--output-def,${CMAKE_CURRENT_BINARY_DIR}/volengine.def`.
  - **`pyproject.toml`:** `version = "0.9.0.dev0"`.

- [ ] **Passo 4: Rodar os testes.** Esperado: os 5 passam.

- [ ] **Passo 5: Verificar no build real (Windows local, ucrt64 primeiro no PATH)**

  Rodar: `python python/scripts/build_native.py && python python/scripts/check_native_deps.py python/src/volengine/_lib/libvolengine.dll`

  Esperado: `ok:`, só DLLs do sistema. `python/src/volengine/_lib/volengine.def` lista os 133 `ve_*`; conferir com
  `grep -c "^ *ve_" python/src/volengine/_lib/volengine.def`.

  Depois, a suíte Python com `VOLENGINE_LIB` apontando para essa DLL, e a suíte C++ do `gcc-release` (a mudança de link
  não pode quebrar nada).

- [ ] **Passo 6: Commit** — `build: native library build for wheels with a static runtime and a dependency check`

### Task 2: Python no uv

**Arquivos:**
- Modificar: `python/pyproject.toml`
- Criar: `python/uv.lock` (gerado)
- Modificar: `.github/workflows/ci.yml` (passos Python)
- Modificar: `CONTRIBUTING.md` (comandos Python)

**Interfaces:**
- Produz:
  - o grupo `dev = ["pytest", "mypy"]`, que substitui o extra `test`;
  - o grupo `bench = ["arch"]`;
  - comandos padrão `uv sync --project python` e `uv run --project python pytest python/tests -q`.

- [ ] **Passo 1:** trocar `[project.optional-dependencies] test` por `[dependency-groups]` (`dev`, `bench`) e
  rodar `uv lock --project python`. O `numpy` continua em `dependencies`.
- [ ] **Passo 2:** verificar localmente.

  Rodar: `uv run --project python --python 3.14 pytest python/tests -q` e depois `--python 3.11`, os dois com
  `VOLENGINE_LIB` apontando para a DLL da Task 1.

  Esperado: a mesma contagem de testes que antes, tudo verde. `test_typing` tem de achar o mypy pelo grupo `dev`.
- [ ] **Passo 3:** no `ci.yml`, trocar os dois `setup-python` + `pip install -e "python[test]"` por
  `astral-sh/setup-uv@v6` com `uv run --project python --python <v> pytest python/tests -q`, para 3.14 e 3.11.
  - O passo "Setup Python 3.14", que serve ao teste `style` do CMake, fica: o `style` usa o Python do runner.
  - `CONTRIBUTING.md`: os comandos Python passam a ser os do uv.
- [ ] **Passo 4: Commit** — `build: manage the Python package with uv`

### Task 3: wheel com tag de plataforma

**Arquivos:**
- Criar: `python/hatch_build.py` (adaptado do FixedWidthParser)
- Criar: `python/scripts/check_wheel_tag.py` (cópia do FixedWidthParser)
- Modificar: `python/pyproject.toml` (`force-include` do `_lib`, hook)
- Testar: `python/tests/test_packaging_scripts.py` (mesmo arquivo da Task 1)

**Interfaces:**
- Consome: `python/src/volengine/_lib/` preenchido pela Task 1.
- Produz:
  - `uv build --wheel python --out-dir dist` gera `volengine-<versão>-py3-none-<tag>.whl`;
  - as funções `glibc_floor(library: bytes) -> str`, `macos_floor(library: bytes) -> str` e
    `platform_tag() -> str` em `hatch_build.py`.

- [ ] **Passo 1: Escrever os testes que falham**

```python
def test_glibc_floor_is_newest_symbol_version():
    assert hook.glibc_floor(b"\0GLIBC_2.17\0GLIBC_2.28\0GLIBC_2.2.5\0") == "2_28"

def test_glibc_floor_defaults_to_2_17():
    assert hook.glibc_floor(b"no versions") == "2_17"

def test_macos_floor_reads_lc_build_version():
    assert hook.macos_floor(mach_o_with_minos(11, 0)) == "11_0"
```

  `mach_o_with_minos(major, minor)` é um auxiliar do próprio teste. Ele monta um cabeçalho Mach-O 64 mínimo com
  `ncmds = 1` em `offset 16` e um `LC_BUILD_VERSION` (`0x32`) em `offset 32`, com `minos = major << 16 | minor << 8`
  em `offset 32 + 12`.

  Cuidado: `2.2.5` não pode virar `2_2` nem ganhar de `2.28`. A regex do FixedWidthParser, `GLIBC_2\.(\d+)`, pega o 2
  de `2.2.5` e funciona; manter.

- [ ] **Passo 2: Rodar e ver falhar.**
- [ ] **Passo 3: Implementar `hatch_build.py`.**
  - Os nomes procurados em `_lib` são `libvolengine.so`, `libvolengine.dylib` e `libvolengine.dll`.
  - No `pyproject.toml`, `exclude` e `force-include` do `_lib`, iguais aos do FixedWidthParser, e
    `[tool.hatch.build.targets.wheel.hooks.custom] path = "hatch_build.py"`.
  - Nomes de arquivo, sem o `.def` nem o `.dll.a`: o wheel não precisa deles, e eles vão só para a release.
- [ ] **Passo 4: Rodar os testes.** Esperado: os 3 passam.
- [ ] **Passo 5: Montar o wheel.**

  Rodar: `uv build --wheel python --out-dir dist && python python/scripts/check_wheel_tag.py dist`

  Esperado: `ok: volengine-0.9.0.dev0-py3-none-win_amd64.whl`.
- [ ] **Passo 6: Instalar num venv limpo e testar (foco 1).**
  - Montar o venv: `uv venv build/wheel-venv`, depois `uv pip install --python build/wheel-venv dist/*.whl pytest
    mypy` (sem `VOLENGINE_LIB` no ambiente).
  - Conferir de onde a biblioteca vem: `build/wheel-venv/Scripts/python -c "import volengine._native as n;
    assert 'site-packages' in str(n.LIBRARY_DIRECTORY), n.LIBRARY_DIRECTORY"`.
  - Rodar a suíte com o pytest do venv, a partir da raiz: `build/wheel-venv/Scripts/python -m pytest python/tests -q
    -p no:cacheprovider`.
  - Esperado: verde. O `python/src` não pode estar no `sys.path`; o pyproject não tem `pythonpath`, então não deve
    estar.
- [ ] **Passo 7: Commit** — `build: platform-tagged wheel from the bundled library`

### Task 4: `wheels.yml`, os 7 wheels em todo push

**Arquivos:**
- Criar: `.github/workflows/wheels.yml`
- Criar: `python/scripts/build_in_container.sh` (os dois containers Linux)

**Interfaces:**
- Consome: `build_native.py`, `check_native_deps.py`, `hatch_build.py`, `check_wheel_tag.py`.
- Produz: o workflow `on: push` (branch master; paths `src/**`, `include/**`, `python/**`, `CMakeLists.txt`,
  `CMakePresets.json`, `.github/workflows/wheels.yml`) e `workflow_call` com a entrada `version` (string, vazia = a
  do repositório). Os artefatos:
  - `wheel-<nome>`: um `.whl` por plataforma;
  - `native-<nome>`: os assets já renomeados conforme as restrições globais.

  `<nome>` é um de `linux-x64`, `linux-arm64`, `linux-musl-x64`, `linux-musl-arm64`, `osx-arm64`, `osx-x64`, `win-x64`.

- [ ] **Passo 1: Escrever o workflow.** Matriz com `name`, `runner`, `container`, `expected_tag`.
  - Os runners são os da tabela do §5.1 da spec.
  - Os jobs Linux rodam `build_in_container.sh` num `docker run`:
    - imagens `quay.io/pypa/manylinux_2_28_<arch>` e `quay.io/pypa/musllinux_1_2_<arch>`;
    - dentro: instalar ninja e uv, rodar `build_native.py`, `check_native_deps.py`, `uv build --wheel` e
      `auditwheel show`, depois testar o wheel instalado como na Task 3, passo 6.
  - macOS usa `env MACOSX_DEPLOYMENT_TARGET: '11.0'`.
  - Windows usa `msys2/setup-msys2` (ucrt64, `mingw-w64-ucrt-x86_64-gcc`, `-cmake`, `-ninja`), roda
    `build_native.py` pelo shell msys2 e `objdump` para a checagem.
  - Cada job:
    1. com `version` não vazia, grava `str(Version(version))` no `pyproject.toml` (a mesma função da Task 5) e passa
       `-DVE_VERSION=<x.y.z>` (sem o sufixo de prerelease);
    2. compila, confere as dependências e monta o wheel;
    3. falha se a tag do wheel não for `expected_tag` (foco 2);
    4. instala o wheel num venv limpo, confere `site-packages` (foco 1) e roda a suíte;
    5. renomeia os assets e sobe os artefatos.
- [ ] **Passo 2: Validar a sintaxe localmente.**

  Rodar: `uvx --from actionlint-py actionlint .github/workflows/wheels.yml`. Esperado: sem erros.
- [ ] **Passo 3: Push (pedir o ok do autor) e acompanhar.**

  Rodar: `gh run watch` no run do `wheels.yml`. Esperado: os 7 jobs verdes.
  - Se o manylinux_2_28 falhar no compilador, aplicar o fallback das restrições globais.
  - Se o macOS falhar por availability, subir o target e registrar.
  - Cada correção é um commit `fix(ci): ...`.
- [ ] **Passo 4: Commit** (antes do push) — `ci: build and test the seven wheels on every push`

### Task 5: `release.yml`, tag publica tudo

**Arquivos:**
- Criar: `.github/workflows/release.yml`
- Criar: `python/scripts/release_version.py`
- Testar: `python/tests/test_release_version.py`

**Interfaces:**
- Consome: `wheels.yml` via `uses: ./.github/workflows/wheels.yml` com `version: ${{ github.ref_name }}`.
- Produz: `release_version.py TAG` imprime a versão PEP 440 e o `x.y.z` do CMake numa linha cada. Funções
  `python_version(tag: str) -> str` e `cmake_version(tag: str) -> str`.

- [ ] **Passo 1: Escrever os testes que falham (foco 4)**

```python
def test_plain_tag():
    assert release_version.python_version("v0.9.0") == "0.9.0"
    assert release_version.cmake_version("v0.9.0") == "0.9.0"

def test_prerelease_tag():
    assert release_version.python_version("v1.0.0-rc.1") == "1.0.0rc1"
    assert release_version.cmake_version("v1.0.0-rc.1") == "1.0.0"

def test_tag_without_v_is_rejected():
    with pytest.raises(ValueError):
        release_version.python_version("1.0.0")
```

- [ ] **Passo 2: Rodar e ver falhar. Passo 3: implementar** com `packaging.version.Version`. `cmake_version` é
  `Version(...).base_version`. **Passo 4: verde.**
- [ ] **Passo 5: Escrever o `release.yml`.**
  - `on: push: tags: ['v*']` e `workflow_dispatch` com a entrada `dry_run` (boolean, padrão true). O dry run roda
    tudo menos os dois passos de publicação.
  - **Jobs:**
    - **`wheels`:** a chamada reutilizável.
    - **`windows-lib`:** windows-latest com `ilammy/msvc-dev-cmd@v1`. Baixa `native-win-x64`, roda
      `lib /def:volengine-native-win-x64.def /machine:x64 /out:volengine-native-win-x64.lib` e sobe o `.lib`.
    - **`publish-github`:**
      - `permissions: contents: write`;
      - baixa `native-*` e o `.lib`;
      - gera o SBOM com `anchore/sbom-action@v0` (`format: spdx-json`, `output-file: volengine.spdx.json`) sobre os
        assets;
      - `softprops/action-gh-release@v2` com `prerelease: ${{ contains(github.ref_name, '-') }}` e
        `fail_on_unmatched_files: true`.
    - **`publish-pypi`:**
      - `needs: publish-github`;
      - `environment: pypi`, se o autor cadastrou com environment (perguntar no passo 7);
      - `permissions: id-token: write`;
      - baixa `wheel-*` para `dist/` e roda `astral-sh/setup-uv@v6` seguido de
        `uv publish --trusted-publishing always dist/*`.
- [ ] **Passo 6: actionlint** nos dois workflows. Esperado: sem erros.
- [ ] **Passo 7: Dry run.**
  - Perguntar ao autor se o Trusted Publisher do PyPI foi cadastrado com um nome de environment. Se sim, usá-lo em
    `publish-pypi`.
  - Depois do push (com o ok dele), rodar `gh workflow run release.yml -f dry_run=true` e `gh run watch`.
  - Esperado: verde, e os artefatos contêm os 7 wheels e os 9 assets nativos (as 7 bibliotecas mais o `.dll.a` e o
    `.lib` do Windows).
- [ ] **Passo 8: Commit** — `ci: release workflow publishing wheels to PyPI and libraries to GitHub`

### Task 6: benchmark do wheel contra o build local

**Arquivos:**
- Criar: `python/benchmarks/bench_library.py`
- Modificar: `docs/performance/benchmarks.md` (seção "Plan 6a")

**Interfaces:**
- Produz: `bench_library.py [--repetitions N] [--out PATH]`.
  - Carga: o `volengine` do ambiente (a biblioteca vem de `VOLENGINE_LIB` ou do `_lib` do wheel).
    - 21 séries simuladas, T = 5000, semente fixa;
    - os ajustes univariados das 12 combinações sGARCH/gjrGARCH/eGARCH(1,1) × norm/std/ged/sstd;
    - um DCC(1,1) mvnorm e um DCC(1,1) mvt sobre os sGARCH-std;
    - todas as threads, erros-padrão desligados.
  - Imprime a mediana e o desvio do tempo total. Com `--out`, grava JSON
    `{"median_s": ..., "stdev_s": ..., "library": ...}`.

- [ ] **Passo 1:** escrever o script sobre a API pública (`Context.fit`, `Context.fit_dcc`; ver
  `python/src/volengine/api.py`). Fazer um aquecimento antes das repetições.
- [ ] **Passo 2: medir no Ryzen 7 5700X, na mesma sessão, alternando A, B, A, B.**
  - A: wheel no venv da Task 3, sem `VOLENGINE_LIB`.
  - B: venv do projeto com `VOLENGINE_LIB=build/gcc-bench/libvolengine.dll` (refazer o `gcc-bench` no mesmo commit).
  - `--repetitions 10`.
  - Esperado: B/A dentro de 1 desvio. Fora disso é bug: comparar as flags de compilação dos dois builds no
    `compile_commands.json`.
- [ ] **Passo 3:** registrar no `benchmarks.md`:
  - a tabela A × B;
  - a máquina;
  - o commit;
  - o toolchain de cada plataforma da release, incluindo o fallback do manylinux, se usado.
- [ ] **Passo 4: Commit** — `bench: the wheel library against the local build`

### Task 7: `v0.9.0`

**Arquivos:**
- Modificar: `CHANGELOG.md` (seção `[0.9.0] - <data>`; o que está em Unreleased desce para ela)
- Modificar: `python/pyproject.toml` (`0.9.0`), `CMakeLists.txt` (padrão do `VE_VERSION` já é 0.9.0)
- Modificar: `README.md` (seção Install: `pip install volengine` e a lista das 7 plataformas)

- [ ] **Passo 1:** CHANGELOG, versão e README. Commit `release: 0.9.0`.
- [ ] **Passo 2: confirmação do autor.** Ela precisa vir explicitamente, porque publicar no PyPI é irreversível: a
  versão não pode ser reenviada.
- [ ] **Passo 3: publicar.** Com o ok, `git push`, `git tag v0.9.0` e `git push origin v0.9.0`, depois
  `gh run watch` no `release.yml`. Esperado: verde, a release `v0.9.0` com os assets e o SBOM, e a `volengine 0.9.0`
  no PyPI.
- [ ] **Passo 4: verificar de fora.**
  - Num venv novo, fora do repositório, sem `VOLENGINE_LIB`: `uv pip install volengine==0.9.0`, depois um ajuste
    sGARCH(1,1) norm de uma série simulada. Esperado: loglik finita e `status == converged`.
  - Passar ao autor um comando `docker run --rm python:3.12-slim ...` (glibc) e outro `python:3.12-alpine` (musl) com
    o mesmo teste, para ele rodar.
- [ ] **Passo 5:** atualizar a memória do roadmap (6a feito, próximo 6b).
