"""Kaggle GPU cell: finish the corpus encode killed at 83%. T4, Internet ON.

The first attempt ran 4.65 h and died with no traceback — the signature of an OOM
kill, not an exception. The cause was in this pipeline: every caption was held in
a list for the whole run and written only at the end. Tokens were written as they
went, so 1470798 of 1780125 images survived; the captions did not exist at all.

This kernel therefore does not redo the 4.65 h. Two facts make that cheap:

  * the survivors are byte-exact — 753048576 B / 512 B, no partial image;
  * everything still missing lives in ONE shard. Cumulative counts are
    90070 / 90022 / 400001 / 400015 / 400002 / 400015 = 1780125, so prep-5 starts
    at 1380110 and the resume point sits 90688 images into it.

So only prep-5 is needed, and it is fetched over HTTP instead of mounted. That is
not a stylistic choice: this token cannot attach kernel outputs as sources (the
API denies kernels.get, which is what source validation needs), while it can read
their output URLs perfectly well. Downloading one 8 GB shard sidesteps the whole
problem.

Captions for prep-0..4 are pre-seeded into the .jsonl before encoding starts, so
that when encode_corpus.py appends prep-5's the file ends up the same length as
the token stream. Its closing check enforces exactly that, and is the only thing
standing between a silent caption/token misalignment and a model trained on
images paired with the wrong words.
"""

import json
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"
TMP = "/kaggle/tmp/prep5"

DONE = 1470798          # images already in the partial token file
SKIP_IN_SHARD = 90688   # DONE - 1380110, the offset inside prep-5
URLS = json.loads('{"tokens": "https://www.kaggleusercontent.com/kf/344193492/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..hDDsHufa0bvjYJ3LztT6UQ.JrcBiKgvibFpgbHNgsYQMouVP8gduLKBZoe_Bq1BzjNZ1cvIqrfLZE9RaRuDvhGONXg1FDWV-XkEpyKMwkUwXm2tltBNyG31GCRd-sYgKo5F0R4jNhgRDgMbPn0wLPU-wvdFRoa6xuVeoJ7buv7opa7FkNvN2eS37yuapgzlCJveibDWnLYo6yRJnHBWgFswk3dD3mhMayq5o8MtHS3-HrZ7m1-cdWOSbAjlqBZPy-oqyINaVpN2wGz_UD5M09YJXq_4makZdoiG3gc9yfTH5py8Q-4cDSeFmp-gV3x0cgXrCW08YRH4hy_x2-2FOeUXyAgW7z7h-MGEyTjrQfFk5m2eNtcP3Hl8Awl1s5DI1oBGx--93k9yu8Rh-tsbIHsH6qv012FiPZUoSNS4yeegoafrzKAilajsI9djsiNVG_FuJXOrVLSant95sA9hC_fjqDU2EgmGhSHmcYpcx7mX__7xLWYTuHE1FovL_sUcKoraKm4AiKpANKrf7bW2qRZkaom19wEsoslSvjGNx8wiT2hgAJMp5K85hyTbfQShJ1E_ghbALLrps4-hlszaOjNoOwJZvGK0KWI-y44C8TjUV4PQoWqUUz489hPjMV6Noz6leG-k-ZMl_4OhTkKULc3B.IXqqSXp7xYwxim1WZR9erw/gwtok_tokens.u16", "caps0": "https://www.kaggleusercontent.com/kf/343448131/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..ookGceZDWhtzQ1ZjGogLgA.YNkezlYwOi8orvIUqz1PmRIVA0Ks2jv4aIZRxX6za7j_GfcjjTsjm5A9D-b4O0-uycZBTaB16LJUw0YCI_Fs1-DttvP-UG4ko44AFKplFmJo15UnkRx92Dtfh0JA_D7AfRaBXCQsz323CsaULL_uFtPfXFNs4llTrwzhQ6gaqexMVkHxVt48kEE4dHAcrwmYUKzDNH-pJhZaotULbkGp5Tem2QatfsU6PnaUQk8HCzZf6yqb9V2tCLlVnqSZ-HeTXKCoR7V1Eg9iQCmKk-1H4w7Nk_Xy0lSDa5O_VCwGXUtZqyRs688wp9PjtRVtEpLE-FF0CAFnxqlqSlXkeT4kBf3NQnaBvjmHFTToVdrClbpTYBbwPDx3PEJ2diM_UCqkULJoYeSlkddxuqtaBC2Ln3gbvYGLUmsnDE3rTZ7zeIHRlLKLUkxx_0T5LjSLauEuVTQ5xNCLlUSIzZN-mhw4dPa9sIIlI_O8Uozh4GJTGVBK1ari6EZKJve3lp4gXv-Jn9cGWEdK411ySgN75VPamnGIEjxzGwAabLMWaciV3bYA5e8s3JfhF6rzOyReIDYrnSbiloXhd8ySGGP9r-kS421A7slmPSzpRfCUJCOyjMx0IoJCpRQsTaGnKifmPp-Z.H4Y9Y-2sYNGslmMwpYeswQ/gweird_captions.json", "caps1": "https://www.kaggleusercontent.com/kf/343448140/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..9C8UsxwIvRBR3MPsxaNLRQ.QGw2n2glDJudbc2T0LCUuMoXgoR0iSpGl2UaTMZF6soe6yn_KLXH3EWKdW5p0-w81YgPd2-FhZYL8Z34EE-DlFnW_hAJrtcJXewqVCXsbT7qliupFpHhQFi7Of5yRm4Ih3N0YcsCYgfokRfmPyJBZb1dXusVniizyZggDKA6Orf1o3S3y7bfnA2LpzCg4tSwWPAkTxIu_f5WGfKsyCXsyfd286JsRJ0QpZjvu0SVRxZPxyIe8IeawLg9ZZuP8wzzI5ONu1Q9G55wW7ByjCEOkdGpKy07VUomO0vPTHUzG1ViKneT6W0RmiD1TvPG9Gp0tTQZ4lBclFmyRkd4jBycCuWKOHanQcQWshS12Ow77DT4pJKxOIje0IBVC2YguJ1TdqL8R4qfm82U01oO4JcTbpWHWxnMpMK_ffyBtBM3oFSITUKThpR0LGBCTHMzRYXy3kDwcLmwjIJyzptGVmjXnFY5nFQeLzBkD2LCQZC1yRqDg9ojeyDtho3_1waNK1OMmhRu67JGJ7NUxkczzJ8UOspL1NYNYSUu7V4SMqGdXFgAufy1ImzfjwYUR-kKJ3cAuCCngKBggFrE6CnRwdtSCqaqUlXLqZTP6zuzZG9AD1FQ7vmWrbTiPydjqqR4Mq1s.SGrwL65gHiiMylO91_38Vw/gweird_captions.json", "caps2": "https://www.kaggleusercontent.com/kf/343562607/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..OQhTo7D0LTFGcC60b0xN6Q.yEmnewKfU9uucOXpQqKmshBa-tR4QnoQtSkB46pwFeo6SeW3Vo1DF9SJD67QuRHDV8JzL2vJFMejDJjdn8yk0IJympVJFj3rH7BYEBH5Zk4Zi5UYzZoIMCL7DUHx7186u07rEc8d6o5JCDa3TrTcBVj0Y238vURHB9CnNEKVscG7GHyX6FEa7ILVf72t70FAJgpD72aQsuGpdPPZvd-j7vy4qJ2gT-5Ivb2A7I2LLNKe8aghWphXmb5KfGVA2uynwPXV9jlUkS4K-KvK3rInA8hJzBGDc7Ys-5lZ-2luML21UYbV_aqMgLzy4J9l6gvxv3FudjprZmgzNhF8YmPzbdnwr5FFd3euyZ--GQykm8_2jtrNoFqFS2xTrUQTZO5tQJA2obWpaZ0DqqEWnQLwfETVVbm4ueXLF0vZ7JUkqfJQDZPHFKe7y6oEVi3a3_BaRIYR0caggNm-r5JiEIHqblUrPXjiqFmqbgnNUIaJNCopM92o0QvbTh4GG02lLrCRdzjfotoWKeNcUsMJnH1Md1zr2RHGaKoYgBiF7V1VgKXdHuqHjGIPMkz4Pv3Y-yhJL8rLu9A54RnDL9dbiIlWGqXHRLIIbiq93l7AXHASh3EbKze990iYTONGuwX4YfvA.OAMCHsoZR62F-ntXJPnVjQ/gweird_captions.json", "caps3": "https://www.kaggleusercontent.com/kf/343562612/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..LEC6LEc90Kv45p1yPwQgsQ.acSYDCnUCuEJeeghKdJZ1UILxg5XOlqMPURBQTvJYWcEYFQOWRFmmN7s3iunXr2xDxdNAwhK88oFGCIJU04RMgUNfb3D2qghQlpzDVOOwI0iSoaPmeswZuTnkaOJXEzqdkbAgAq-gaL75g1RPZquGFoTr6qBbnoZ7g-QkDXWXUvUnMwACbWH4kh4MiRSSIhtJre4-9glg_6WH1uXKefguEiMHryHGtZjfs5-HPOM1-qnY1EHBcIKg5aF-uFgv4owusqxBm4-QUkJ9kfugUXCiF9qooULp3X_iRDbdQH3TUmNYdkpsVl1W8AL_2EmudgsVzuOhuC3Nt3BLhWbV-SzK3hNX4yMBh7S-JNXNgBM1BtwnCDS7p2liynIq0XP6fDytSaV2KEmAOCKHlRw3burb9eRocoFZ0CLqEkWVNwl8fwm18JnCtshM1aSCYN93wWCfSyiSv2kMftTWFcyKFLx7le6gvy-3HuPil8woJWg0krzO1ebEuVI5cWCU2B46oe7N2JtdRjZCfS30VbRzzO8fJU0dMul2THAT8dCWJ08qPNQ4BWAYSOS3ojDk5NM_ZH0d281TE-0_sm42zJBiYCcGgLxoIBbOJeDnnIvse8UYLPqRrUcgtcbQDT3VZEwvptT.YYDjsVrKISVXkReWOQZn8A/gweird_captions.json", "caps4": "https://www.kaggleusercontent.com/kf/343562621/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..lsWAZQ6VCqWeCumzixdwRA.JvLLvdOz7wbmKXMb2wnPbxRwBjJGj5Ucq_zYuojq-zAXlJ8xhxbwf9QVyW70AtRsVgTdOKaQRo6Bqa3Nh2zVzmH13Y0c-rjGkJn2-5ZrJ2xZDzMiaFPx4hltE1hl5_DJ8-hnV1NHz7OCV2BgFvc7udIQQV7zNnvLUhEvPs-_uO2m7Jdyn0bCIcckta3Pbto_m0qgmOZWcpD8N8yVIRsflQ2fQaKRkHCj0LGjX_m1VSj3qWw67qeLX_rgLRnk2GEDIgMFKqdlMWmN-cSedyx8r2vxdwYnAgMUuPPDD81B5CtvRbfxsMEIklRJMUMi51lnC597QwHvBlT5fBaMlXN6jbVIevDUWh9-c7s-SNVZdseE_gDQYRetmJsjsuzzz1UI4TSwiqkCr9Y_A2MVkcbEQC6_Hbx51CBq_sGJUfelenS7fp4EgwIlj4R8odRo8edx3OX7u94lXIMZRpBr2jRM5FNfXD93T8H_K9deYSF1JZAsYRwHzy16bsKe76QB2Yo0Hw0lHhUMYybGOobHjsvKrVZ75axEpqTZz5zBQocf65v5gHu31-dN9hZRXhqqqnf6y20A-Iju05XRA_JPp3U76XLnDjVoIJex95ho_P9TK4Vsz8THIt-i77VVR1zVUEL_.laE9UNpqzsEAKyC5_9W07A/gweird_captions.json", "meta5": "https://www.kaggleusercontent.com/kf/343562626/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mT8HEU01DKsnTEkMrUZiyA.MaCZqutzxMsX2Ox52-PHSlo6OCY3ZP2iUYBvcPCxklVNsOHVFAO4C7UPHb3k8vNJrjKoyf8GzATtD8mRB5vAMJD-NCNXC4LliG_ULowou9fhXH8F4JKdbTYRYBfFH3cKYLFg32gQYvo07eTWO9xDJtLvu4XFJ1ktZ5DZVoZVmocKnJkjY-7EI8qGbZ1KRqoTZbXXJyu_qsXj5QHedXOu-Wp6i8dlmybGvgP0obFrzramcuIBcZQu4BzCow2xpJv6As5S3wtVPNic5SNFXRN-jAQHdWI_nhxIDpeILtWTA-EA3NudreU9_5d5J1FisqBb5SQ8m5kXUnt5lgJBEh6IHVjDd9MQtwAIBFkJJhmhYPKxmWLBtjB2CUV-igAWFqPSks6w8_cJdGkz6C29RsyiNLVWyV-qthR8hFfQJJ9V0S-hy7BNvBugawPMcLPlWirVD1bPXO5fd-f6Dc-bOjotiB1oVAvCq2x-iVNcIjHT_5cJbKIfviOLemG5kYqsE3xtlazAEuwa5wM5FxzqgiBmwvAoQky45LbZO8iAeCVRQY-X_opz-42wbq4Gp-l3QlR-ywKZpDRmrWHFWr7w4yKzJf6y5CKn3Jp1ZqyCO_N8hxMZyRQpLC3kjnn_FWz5wp01.CrCHkXyvkfdCyel7mtp-6w/gweird_meta.json", "offs5": "https://www.kaggleusercontent.com/kf/343562626/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..gH2aUVGfc_LddCx3Alw4nw.ese6vM10EVD_WY6JFrM-ArAWmTGRqPzrmdMi7Em4F038P7MtSFSNVc5IuG36C8Ua4pR3xQQbxdWOIE4zpSNezCYPCWgUC80eYDt4yOUChQ81wJCG3JaZ8zIAFC2hxl13KStjIXaBo2zUNQ0A1K9iptiwvzDRSd3zvdOctNTGC-jGr2xF_xxQCc5CfJJnZUq3ykc3d-gHZnjxs4Pez0qBD0Gl-gR1uru9UVqVyrIEQOKUS7f0LDSKhITZ9cDC0KU4qXhrxNKIj9CeWUgY4OqnDf_WNVPjrfk9QSz3iJfLBO3lXfOmyI5YC5QpBTC-Y3HYBckWATyGio9cEhvfyOHbgT0bp08JA5AyStly5wpzTB_y0fqIz-egooPi0OUMrQXRSOhol89qVXLyawfZe1_VqCdVmLVhN7RmDVnRscAGJRfduP7hkSX5Sg8uKoJZB5Zf1uO6sUsshQvvJw6mwkAYf5HgwaefIFz2nVZMQXcBb265by_6szVjcSaLcmIJsKhde1j9vR0lgBJBbFHasG1Ax77XQ7FX6eN2Gf61Oq0-ZJxDcOgIAvc7dXEQeuPWd4WZ3uwRdtUlKgkzUkUkpGfkOz6S8aN_gg8hfFALG7ABxwALJjRZXB8UJ4zHVnxbxaDV.dpesahX-emWruo3gpCMSRQ/gweird_offsets.json", "caps5": "https://www.kaggleusercontent.com/kf/343562626/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..T4QeIIp8DWXxisicJLgZWg.pJLJFwH7gvmhsZgV47zrGuoWTBoNEpdXJT2yh-GVPHK3jCb4VcDoQIZiZ5BZ3_wc131tnWtY0ULdN9hPMmInsimnZpGzojwOp3g7MV3g9hsTHN-XObmiv7RkLe1b9Upf2GR-1qaDwWur_LpTscHZ9-505iku2n28-OojGUjR0CwkSAXMoFf2r6rnYVivn1tKBkoCbYQWKx0jfq6MuYil7prW-h64Tykl8SH4BFHd8v4wXUBiDT6CninQ9eLDjSCba9DoFk1UYYCz19pvhr_nQjCPy6thky7jdWfLe0YwOuWcneTZYeqUcurP1dSyjmYRGdHiyUkwbl4NlkL4tWqdEJLENJUyM7HKFPQCarT7a2Q7DbBkF4IHFKiop6hFH6Kzak5NFpkxOEz7p8ZtRSkN7uD3iHtOxKbvH9Z_LJHfsjRs7gr9islSloQsdIgOQcOuhJauTy1NzyLJ0Qd-4VwV4dp-22PLrY8hlRCFBzFJHAxdpKzl7icg1RHp9CCr2hduK33sTRQnDOS-XmYUnKHJ-2eRW37qQuh2bvKZJ-2bZsZ8BOY_xy1j5NHnNC7Fzg6JyuRzZVMw9e25heWJsMRFj3kNvoUgF0roqXwNSXdVw-GaWPFeLE41KZ3EzzmeKYJi.-SQB-BHL85oNyF24tMzwyw/gweird_captions.json", "img5": "https://www.kaggleusercontent.com/kf/343562626/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..6R5Jp0XAh8TnGHPtqxnFRA.6h-DOQtVmlHlN362ACwSCIldNaj8iw1MKSFjEti7tP18mW_DPOa83ow0o9DHyg4Wh14-3cKpCZdkc_CiInkjQn0AHyq0TIKmng2T3W48eYViuSsHrc26r9XpwVYlB1GlGnrQQ9hT5hFGT-mIh7WYRU1uHk1Yynd6kINDOBADSrD-7gju7qvIigfrqV31vlRBlsVP9em7sRuioxMnO873HomfVdL9L9GGWnTjWgyRz0j9TsVf4vwJzeniYEFsxI99zhYNXqw0GKEOhhhi9oI5PLSNwhEiuFV55Ei7mMLMhETbTyyMGKwWU1zhI5yjsRKH4sD8P7r_T1wqN5-SgK0ZXg_Lm9fJCdpoX_IMOiza3qCJewSsiRRFCQDCfNJrPSP3AbpirhMlIjkPQVR-z9T_u0e9TbIr1hbo5IQIEOzPeC8zCJ6Eiz3c3q5K1dZOodbVs9B0BcA5cRofG2gqu-Od0XCcwH8yg3-ijPpUq-FHSl7bDOUVtsAwj1ds_jJuubegC9ysFxJUiiCV5HYu9veQYwkIpvqx8A6w3U5DxWkQXCbQDO8ExgZ98GxN2rvbQcNDDtr6E7bxM8_KrXf0qVdy33WiXO3bAU8Abs-7Nedi1gFfeMCtz2Wobv9rxyxMa2UI.KD-mW6zZO94gX5CEQl86Ig/gweird_images.jpgbin", "vqvae": "https://www.kaggleusercontent.com/kf/344091927/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..jjXKxRLxKJiRht4d09L50A.-BszFDff4HVxlvcXv8vOnExL1hLd-xom3XXeP5h_ORdS3yC_SMBh_olH19QfxsM6dkGo5f_vk60nTNnvJztGxHA1FUKwlI9WfhtAjIU0lBS-63lV3DZyqmMwz1D-lptezO9AH3d8FAVjwo4wrIhEZQjkxBrQqGo96IUmICg5X6aQ4CpkKnDRa-D1MVh63MR6G0WVq6EeBg_BaOvS_go_shTrpXwl4n9CAdrU5d0aT2Ez4VH-xvrlkeU1ywWAB3Ml0kTdM764l1YHjoy7jbWFdwywE8EKptNnaJoi289fB1DH_CD3QMSzpdvR9-mAshskQtImXiprryDswnbUF-3wVD3v7GOpMTiMSsMh1jChVA9hxasKS8ZHxaejdMGnqJTL9qEnuMqfg0I4DP4itnmuns0-vs5q_Gx9s51K9gs8wn-9MBP0i5_GFZzJMkpv7t_QnLKlxQPqbODt-mvbksHg6fVYb5gX9y8UTjKbgRt56P4sPxrmW0eVUMrEdxrRs6liOzk89x-jJRN_DmqaqnY5n7nZAKufK8ZW983iYrtlIV6tZfPTQ0C77goetYqcyOGdxs6rinoFPao1aDmBy2UI8ds20MXji3WorsMSdmU12e0AQVjNZrD9wLviG1-nn3K9.0epzHHu_dUFTJXkQRb73hQ/run/vqvae.pt"}')

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")
os.makedirs(TMP, exist_ok=True)


def grab(key, dest, tries=8):
    """Download with resume. The first attempt at the 8 GB shard died on
    `curl: (92) HTTP/2 stream not closed cleanly` after several minutes, and
    --retry did not save it: that flag covers failures establishing the transfer,
    not one that breaks mid-stream, and without --continue-at a retry would
    restart from byte zero anyway. So: HTTP/1.1, which is what avoids the stream
    error on kaggleusercontent, plus range resume, plus a stall detector."""
    if key not in URLS:
        raise SystemExit(f"brak linku dla {key}")
    for attempt in range(1, tries + 1):
        r = subprocess.run(["curl", "-sSL", "--http1.1", "--continue-at", "-",
                            "--retry", "5", "--retry-all-errors", "--retry-delay", "5",
                            "--speed-limit", "10000", "--speed-time", "60",
                            "-o", dest, URLS[key]])
        if r.returncode == 0:
            break
        have = os.path.getsize(dest) / 1e6 if os.path.exists(dest) else 0
        print(f"  {key}: proba {attempt}/{tries} przerwana (curl {r.returncode}), "
              f"mam {have:.0f} MB, wznawiam", flush=True)
    else:
        raise SystemExit(f"{key}: nie udalo sie pobrac w {tries} probach")
    print(f"  {key}: {os.path.getsize(dest)/1e6:.1f} MB", flush=True)
    return dest


print("pobieram czesciowe tokeny...", flush=True)
tok = grab("tokens", f"{WORK}/gwtok_tokens.u16")
size, want = os.path.getsize(tok), DONE * 256 * 2
if size != want:
    raise SystemExit(f"czesciowy plik ma {size} B, oczekiwano {want} B — "
                     f"link wygasl albo wskazuje co innego")

print("skladam podpisy shardow 0-4...", flush=True)
seeded = 0
with open(f"{WORK}/gwtok_captions.jsonl", "w") as out:
    for i in range(5):
        p = grab(f"caps{i}", f"{TMP}/caps{i}.json")
        for c in json.load(open(p)):
            out.write(json.dumps(c, ensure_ascii=False) + "\n")
            seeded += 1
        os.remove(p)
print(f"  {seeded} podpisow (oczekiwano 1380110)", flush=True)
if seeded != 1380110:
    raise SystemExit(f"shardy 0-4 daja {seeded} podpisow, nie 1380110")

print("pobieram shard prep-5...", flush=True)
for key, name in [("meta5", "gweird_meta.json"), ("offs5", "gweird_offsets.json"),
                  ("caps5", "gweird_captions.json"), ("img5", "gweird_images.jpgbin")]:
    grab(key, f"{TMP}/{name}")
ckpt = grab("vqvae", f"{WORK}/vqvae.pt")

subprocess.run([sys.executable, "train/encode_corpus.py",
                "--data", f"{TMP}/gweird", "--ckpt", ckpt,
                "--out-prefix", f"{WORK}/gwtok", "--batch", "64",
                "--skip", str(SKIP_IN_SHARD)], check=True)

os.remove(ckpt)
for f in sorted(os.listdir(WORK)):
    p = f"{WORK}/{f}"
    if os.path.isfile(p):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB", flush=True)
