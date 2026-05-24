import os
import sys

# Dynamic path resolution to import asr_postprocess from asr/src
sys.path.append(os.path.join(os.path.dirname(__file__), "../asr/src"))
from asr_postprocess import digits_to_words

def test_digits_to_words():
    # 1. Numerics
    assert digits_to_words("123") == "one hundred twenty three"
    assert digits_to_words("123rd") == "one hundred twenty third"
    assert digits_to_words("0.8.4") == "zero eight four"
    assert digits_to_words("0.8") == "zero point eight"
    assert digits_to_words("0.3%") == "zero point three percent"
    assert digits_to_words("40%") == "forty percent"
    assert digits_to_words("decimal2") == "decimal two"
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
    assert digits_to_words("Sionite") == "Cyanite"
    assert digits_to_words("sanite") == "cyanite"
    assert digits_to_words("sinide") == "cyanite"

    # 4. Renhwa variants
    assert digits_to_words("renwa") == "renhwa"
    assert digits_to_words("Ren Ha") == "Renhwa"
    assert digits_to_words("Renha") == "Renhwa"
    assert digits_to_words("Renoa") == "Renhwa"

    # 5. New Mewan
    assert digits_to_words("New Mu1") == "New Mewan"
    assert digits_to_words("New Mi One") == "New Mewan"
    assert digits_to_words("New Maven") == "New Mewan"
    assert digits_to_words("New Niwan") == "New Mewan"
    assert digits_to_words("numiwan") == "new mewan"
    assert digits_to_words("maven") == "mewan"
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
    assert digits_to_words("Devikauranya") == "Devika Oranyan"

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
    assert digits_to_words("Cape Teda") == "Cape Tidak"
    assert digits_to_words("Cape Iraq") == "Cape Tidak"
    assert digits_to_words("tidakran") == "tidak run"

    # 13. Blackshore
    assert digits_to_words("black shore") == "blackshore"

    # 14. Veyanova
    assert digits_to_words("vayanova") == "veyanova"
    assert digits_to_words("Vyanova's") == "Veyanova's"
    assert digits_to_words("vianova") == "veyanova"
    assert digits_to_words("Vaianova") == "Veyanova"
    assert digits_to_words("Vayanawa") == "Veyanova"

    # 15. Sarento (extra)
    assert digits_to_words("Sarrento") == "Sarento"
    assert digits_to_words("serento") == "sarento"
    assert digits_to_words("Sarantu") == "Sarento"
    assert digits_to_words("Sarantosite") == "Sarento site"

    # 16. Kashikari (extra)
    assert digits_to_words("Kashigari's") == "Kashikari's"
    assert digits_to_words("kashkari") == "kashikari"

    # 17. Tavenport (extra)
    assert digits_to_words("Tavernport") == "Tavenport"
    assert digits_to_words("davenport") == "tavenport"

    # 18. Park Soo-Hyun (extra)
    assert digits_to_words("Park Suzanne") == "Park Soo-Hyun"
    assert digits_to_words("Park Suhyon") == "Park Soo-Hyun"
    assert digits_to_words("Park Suhyin") == "Park Soo-Hyun"
    assert digits_to_words("Paksu Hyon") == "Park Soo-Hyun"
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
    assert digits_to_words("Zone nine Maritime") == "Zonnon Maritime"
    assert digits_to_words("Sonon") == "Zonnon"
    assert digits_to_words("Zonon's") == "Zonnon's"
    assert digits_to_words("zondun's") == "zonnon's"

    # 22. Caulfield
    assert digits_to_words("Coalfield") == "Caulfield"
    assert digits_to_words("callfield") == "caulfield"
    assert digits_to_words("Coalfield's") == "Caulfield's"
    assert digits_to_words("callfields") == "caulfields"
    assert digits_to_words("Coffield's") == "Caulfield's"

    # 23. Canian
    assert digits_to_words("kanyan") == "canian"
    assert digits_to_words("canaanian") == "canian"
    assert digits_to_words("Canadian") == "Canian"
    assert digits_to_words("Kenya") == "Cania"
    assert digits_to_words("Kleros") == "Clairos"

    # 24. Hegemony
    assert digits_to_words("hegemoni") == "hegemony"
    assert digits_to_words("Hegmoni") == "Hegemony"

    # 25. Sharpsea
    assert digits_to_words("sharp sea") == "sharpsea"
    assert digits_to_words("SHARP C BLOCK") == "SHARPSEA BLOC"
    assert digits_to_words("sharp-c routes") == "sharpsea routes"

    # 26. Nyari
    assert digits_to_words("niari") == "nyari"
    assert digits_to_words("niyari") == "nyari"
    assert digits_to_words("Niari") == "Nyari"

    # 27. Dreamer
    assert digits_to_words("streamer") == "dreamer"
    assert digits_to_words("Streamers") == "Dreamers"

    # 28. Fullwalker
    assert digits_to_words("full walker") == "fullwalker"
    assert digits_to_words("pull walkers") == "fullwalkers"

    # 29. Edgedancer
    assert digits_to_words("edge dancer") == "edgedancer"
    assert digits_to_words("edgedenser") == "edgedancer"
    assert digits_to_words("Adjudancers") == "Edgedancers"

    # 30. Floodwall
    assert digits_to_words("flood wall") == "floodwall"
    assert digits_to_words("flood walls") == "floodwalls"

    # 31. TEC / tech
    assert digits_to_words("tech command") == "tec command"
    assert digits_to_words("Tech command") == "TEC command"
    assert digits_to_words("Tekki's politics") == "TEC's politics"
    assert digits_to_words("for tech") == "for tec"
    assert digits_to_words("for Tech") == "for TEC"
    assert digits_to_words("tech team") == "tech team"

    # 32. CYPHER / cipher
    assert digits_to_words("cipher requires") == "cypher requires"
    assert digits_to_words("Cipher requires") == "Cypher requires"
    assert digits_to_words("Cipher's acknowledgement") == "Cypher's acknowledgement"
    assert digits_to_words("Ciphers left") == "Cypher left"
    assert digits_to_words("give cipher") == "give cypher"
    assert digits_to_words("give Cipher") == "give Cypher"
    assert digits_to_words("cascade cipher") == "cascade cipher"

    # 33. Bloc / block
    assert digits_to_words("block tensions") == "bloc tensions"
    assert digits_to_words("Block tensions") == "Bloc tensions"
    assert digits_to_words("Accommodationist block") == "Accommodationist bloc"
    assert digits_to_words("Accommodationist Block") == "Accommodationist Bloc"
    assert digits_to_words("road block") == "road block"

    # 34. Cleanup artifacts
    assert digits_to_words("Uh") == ""
    assert digits_to_words("TheCUBE's angle") == "The CUBE's angle"
    assert digits_to_words("firstdreamer") == "first dreamer"
    assert digits_to_words("synchronization") == "synchronisation"
    assert digits_to_words("Maritime Defence") == "Maritime Defense"
    assert digits_to_words("armored support") == "armoured support"

    print("All unit tests passed successfully!")

if __name__ == "__main__":
    test_digits_to_words()
