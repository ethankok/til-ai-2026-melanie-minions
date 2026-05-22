import os
import sys

# Dynamic path resolution to import asr_postprocess from asr/src
sys.path.append(os.path.join(os.path.dirname(__file__), "../asr/src"))
from asr_postprocess import digits_to_words

def test_digits_to_words():
    # 1. Numerics
    assert digits_to_words("123") == "one hundred twenty three"
    assert digits_to_words("123rd") == "one hundred twenty threeth"
    assert digits_to_words("0.8.4") == "zero eight four"
    assert digits_to_words("0.8") == "zero point eight"
    assert digits_to_words("0900") == "zero nine hundred"
    assert digits_to_words("0430") == "zero four thirty"
    assert digits_to_words("2300") == "twenty three hundred"
    assert digits_to_words("1,234") == "one two thirty four"
    assert digits_to_words("1234") == "one two three four"

    # 2. Sorrento/Sorento -> Sarento
    assert digits_to_words("Sorrento") == "Sarento"
    assert digits_to_words("sorrento") == "sarento"
    assert digits_to_words("SORRENTO") == "SARENTO"
    assert digits_to_words("Sorrentos") == "Sarentos"

    # 3. Cyanide/Syanite -> Cyanite
    assert digits_to_words("cyanide") == "cyanite"
    assert digits_to_words("Syanite") == "Cyanite"
    assert digits_to_words("sanite") == "cyanite"
    assert digits_to_words("sinide") == "cyanite"

    # 4. Renhwa variants
    assert digits_to_words("renwa") == "renhwa"
    assert digits_to_words("Ren Ha") == "Renhwa"
    assert digits_to_words("Renha") == "Renhwa"

    # 5. New Mewan
    assert digits_to_words("New Mu1") == "New Mewan"
    assert digits_to_words("New Mi One") == "New Mewan"
    assert digits_to_words("numiwan") == "new mewan"
    assert digits_to_words("New Muvan's") == "New Mewan's"

    # 6. Phyrexis
    assert digits_to_words("perex") == "phyrexis"
    assert digits_to_words("pyrexis") == "phyrexis"
    assert digits_to_words("Pyrex's") == "Phyrexis's"

    # 7. Kestrelian
    assert digits_to_words("castralian") == "kestrelian"
    assert digits_to_words("Kestrillian") == "Kestrelian"
    assert digits_to_words("Kesrelian") == "Kestrelian"

    # 8. Oyelaran
    assert digits_to_words("takeshi oilaran") == "takeshi oyelaran"
    assert digits_to_words("Ada olrn") == "Ada oyelaran"
    assert digits_to_words("Olrn") == "Oyelaran"

    # 9. Oranyan
    assert digits_to_words("devika aranyan") == "devika oranyan"
    assert digits_to_words("Divikauranyan") == "Devika Oranyan"

    # 10. Soo-Hyun
    assert digits_to_words("Park Soo Hyun") == "Park Soo-Hyun"
    assert digits_to_words("park suyan") == "park soo-hyun"
    assert digits_to_words("soo hyun") == "soo-hyun"
    assert digits_to_words("Suyan") == "Soo-Hyun"

    # 11. Sim Jiahong
    assert digits_to_words("Sim jahong") == "Sim Jiahong"
    assert digits_to_words("Jahong") == "Jiahong"

    # 12. Tidak
    assert digits_to_words("tedak") == "tidak"
    assert digits_to_words("tidakran") == "tidak run"

    # 13. Blackshore
    assert digits_to_words("black shore") == "blackshore"

    # 14. Veyanova
    assert digits_to_words("vayanova") == "veyanova"
    assert digits_to_words("Vyanova's") == "Veyanova's"
    assert digits_to_words("vianova") == "veyanova"
    assert digits_to_words("Vaianova") == "Veyanova"

    # 15. Sarento (extra)
    assert digits_to_words("Sarrento") == "Sarento"
    assert digits_to_words("serento") == "sarento"

    # 16. Kashikari (extra)
    assert digits_to_words("Kashigari's") == "Kashikari's"
    assert digits_to_words("kashkari") == "kashikari"

    # 17. Tavenport (extra)
    assert digits_to_words("Tavernport") == "Tavenport"
    assert digits_to_words("davenport") == "tavenport"

    # 18. Park Soo-Hyun (extra)
    assert digits_to_words("Park Suzanne") == "Park Soo-Hyun"
    assert digits_to_words("Park Suhyon") == "Park Soo-Hyun"
    assert digits_to_words("Park Shohyan") == "Park Soo-Hyun"
    assert digits_to_words("suzanne") == "soo-hyun"

    # 19. Ashcastle / Tell Ashcastle
    assert digits_to_words("Ash Castle") == "Ashcastle"
    assert digits_to_words("Del Ash Castle") == "Tell Ashcastle"
    assert digits_to_words("Skel Ash castle") == "Tell Ashcastle"
    assert digits_to_words("delashcastle") == "tell ashcastle"
    assert digits_to_words("is Ash Castle") == "is Ashcastle"
    assert digits_to_words("placing Ash Castle") == "placing Ashcastle"
    assert digits_to_words("is Ashcastle") == "is Ashcastle"
    assert digits_to_words("del-ash-castle") == "tell ashcastle"

    # 20. Devika Oranyan (extra)
    assert digits_to_words("De Vika Oranyan") == "Devika Oranyan"
    assert digits_to_words("Davika Oranyan") == "Devika Oranyan"
    assert digits_to_words("Devika Uranyan") == "Devika Oranyan"
    assert digits_to_words("Devika Auranyan") == "Devika Oranyan"
    assert digits_to_words("Devika Origins") == "Devika Oranyan"
    assert digits_to_words("uranyan") == "oranyan"
    assert digits_to_words("auranyan") == "oranyan"

    # 21. Zonnon
    assert digits_to_words("Zonan") == "Zonnon"
    assert digits_to_words("zonon") == "zonnon"
    assert digits_to_words("ZONON") == "ZONNON"
    assert digits_to_words("Zonanun") == "Zonnon"
    assert digits_to_words("Zonal") == "Zonnon"
    assert digits_to_words("Zonon's") == "Zonnon's"
    assert digits_to_words("zondun's") == "zonnon's"

    # 22. Caulfield
    assert digits_to_words("Coalfield") == "Caulfield"
    assert digits_to_words("callfield") == "caulfield"
    assert digits_to_words("Coalfield's") == "Caulfield's"
    assert digits_to_words("callfields") == "caulfields"
    assert digits_to_words("Coffield's") == "Caulfield's"

    print("All unit tests passed successfully!")

if __name__ == "__main__":
    test_digits_to_words()
