# Super resolution

Uruchamiaj notebooki od góry, w tym samym środowisku co dotychczas.

1. `01_data_preparation.ipynb` zapisuje lokalny podział WikiArt raz. Ponowne uruchomienie wykorzystuje gotowy zapis.
2. `02_super_resolution.ipynb` trenuje listę modeli kolejno. W `MODELS` wpisujesz nazwę, architekturę i jej parametry, a w `TRAINING` wspólne ustawienia. Strata to zawsze MSE. Każdy model otrzymuje osobny katalog eksperymentu.
3. `03_super_resolution_benchmark.ipynb` porównuje wybrane modele na pełnej walidacji.
4. `04_super_resolution_test.ipynb` ocenia wybrane modele na pełnym teście, po zakończeniu wyboru ustawień.
5. `05_perform_sr.ipynb` powiększa zdjęcie użytkownika bez dostępu do WikiArt.

Katalog modelu wskazujesz np. przez `EXPERIMENTS_PATH / "wide_128_x2_mse_data_czas"`. W notebookach 03 i 04 dodajesz takie ścieżki do `RUN_PATHS`, a w 05 ustawiasz pojedynczy `RUN_PATH`.

Ścieżki budujemy przez `pathlib` względem automatycznie wykrytego katalogu projektu. Komunikaty, nowe konfiguracje i raporty pokazują ścieżki względem projektu, ze znakiem `/` niezależnie od systemu. Dla danych spoza repozytorium zapisujemy tylko nazwę katalogu; pole `dataset_path` jest opisowe, a faktyczną lokalizację wskazuje `DATASET_PATH` w notebooku. Zgodność danych sprawdzamy przez `partition_info`. Dotychczasowe konfiguracje i checkpointy pozostają czytelne bez ich modyfikacji.

Przed publikacją wyczyść wyniki komórek notebooków, ponieważ komunikaty bibliotek i tracebacki mogą zawierać lokalne ścieżki. Katalogi danych, wag i wyników pozostają ignorowane przez Git.

Do dotrenowania ustaw w 02 `RUN_MODE = "resume"`, uzupełnij `RESUME_PATHS` i zwiększ `EPOCHS`. Jest to łączny cel, np. 150 po zapisanych 100 epokach oznacza jeszcze 50 epok. Pozostałe ustawienia pochodzą z zapisanego eksperymentu.

W 02 ustawiasz `NUM_WORKERS`, domyślnie 2; wartość 0 wyłącza procesy pomocnicze. Ustawienie trafia do obu loaderów treningowych i konfiguracji eksperymentu. Przy wznowieniu obowiązuje zapisana liczba workerów. Procesy są odtwarzane co epokę, aby zachować powtarzalność losowania po wznowieniu; przy małych zbiorach koszt ich uruchamiania może przeważyć nad zyskiem z równoległości.

W 03 i 04 ustawiasz niezależne `NUM_WORKERS`, również domyślnie 2. Ocena zachowuje procesy między przebiegiem bicubic i kolejnymi modelami, ponieważ dane nie mają losowej augmentacji. Liczba workerów trafia do raportu, a mierzony czas forward nadal nie obejmuje wczytywania danych. Notebooki 01 i 05 nie potrzebują tej opcji.

Seed treningu nie zmienia utrwalonego podziału danych. Limity wybierają początek zapisanych części; pełne porównania w 03 i 04 nie korzystają z tych limitów.

Checkpointy i historia pozostają w katalogu eksperymentu. Raporty trafiają do jego folderów `validation` i `test`, a zdjęcia użytkownika do `predictions`. Zbiorcze porównania znajdują się w `data/sr_results`. Ponowne wykonanie oceny zastępuje jej raporty, zachowując checkpointy.

Porównujemy pełne RGB bez wycinania brzegów. Strata MSE jest liczona przed ograniczeniem wartości, metryki MSE/PSNR/SSIM po ograniczeniu do [0, 1], a PSNR jest średnią po obrazach. Czas inferencji oznacza amortyzowany czas forward na obraz przy podanym batch size, po trzech rozgrzewkach, bez transferów i metryk; nie jest czasem całej obsługi pojedynczego zdjęcia.

Funkcje znajdują się w `src/restore_pipeline/super_resolution`: `data`, `models`, `training`, `checkpoints`, `evaluation`, `visualization` i `inference`. Pomocniczy `paths` formatuje ścieżki do komunikatów i metadanych. Nowa architektura wymaga osobnej klasy oraz przypadku w `build_model`; dotychczasowych wersji nie zmieniamy w sposób niezgodny z ich wagami.

`espcn1_v1` to dotychczasowy model z dwiema konwolucjami i parametrem `hidden_channels`. `espcn3_v1` odtwarza układ warstw końcowego modelu z [publikacji ESPCN](https://openaccess.thecvf.com/content_cvpr_2016/html/Shi_Real-Time_Single_Image_CVPR_2016_paper.html): konwolucje 5×5, 3×3, 3×3, kanały 64 → 32 oraz tanh po dwóch pierwszych konwolucjach. Zachowujemy RGB i padding utrzymujący rozmiar, więc nie odtwarzamy całego protokołu eksperymentalnego autorów. W `MODELS` wybierz `{"name": "espcn3", "architecture": "espcn3_v1", "params": {}}`; liczby kanałów są stałe, a skala pochodzi ze wspólnego `TRAINING`.
