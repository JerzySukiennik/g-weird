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
URLS = json.loads('{"tokens": "https://www.kaggleusercontent.com/kf/344193492/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..d-TH9SXQUmrlVAMHyU7XfQ.UGmRzZ4NTDq2LdmI0NFhDRV0NVvVskYfmzSRiwrRmtE0OMtyiCEZXiGLEi6vtDEf7ZlMcj4NCJk45YlnD78UO2sKgI7lOwor-6ulslSAWaQ7r0Hf7I4xf8VEMuDfl7n-2qQeFuHLVvTAo9DetPH2qgIw5Hhw0I5lrW2-sNg4YfKQ71axNcdRZOXk1OtE2lfVy71bScOA9zG5XtZzWOD_cpqZdV3JyoC_JYAGLbMa0oXLQLz-3S1_jXFh-QMChd_CUZ3z9DG9BGPXNhEtdEflyflZBC8quq0hGq71ZjFP-ZfSH-TCECngY3OqERC0rubO8mly3R8a2T9g0Y6gOGwaOafa_YCAJJP0D71vHM6rj7Kl-DqCmD9aIW_kBG0AfQWmSRweM-BpXevHH44IKKBpwAWGT64DLCBvGARNNam3ZPW4gRN4chwxFBu869jvVRnr12nZ7yKxOgB6-EiDwQdyNCbBOqiwbE--GAHeuC_YwoTla8Ane670AE5VD8AIPdO83Vp6QlsXlQzLdEOSgqE--ezihtq2_z77ikKvqs27gsiCABlgLBlQfdSjHY-Wz1K5frsXQ-q8272ZbfECYSWk54Tz8b3NQuhkbBpZv59yZzFYSnijTf17uqsSVjja9hOy.Om87OQw-5zGKVfHXZcyvkQ/gwtok_tokens.u16", "caps0": "https://www.kaggleusercontent.com/kf/343448131/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..wAMjFmGfBiOjH0YJGCKTBQ.mIy1UaiT5zrhR2Ucuh8B1xePQf2Awa_zQsR6XCOkYAedbBzONQxH3L8nkYenTknmyWmaDHK-ZGLmf3iCoBHsiId8t7aV3a7SYpLxebLDMa9HAskVAbfOpdt7o5wfPJvi6_EQjrr7YiyBqiDZz4Fxy5mK1aOJyO0o0VL4pEuSOvA1m37KoZEYPHJa0H8nu7Mn1RNOD5RO6Fyi39UU23fdCVIUImmUhm1hhAZ7fUpdvFgQeWLXuawGgtDqhmVljwzWnhRrKO5sv2WDPqT1OSbaK_mRnMRFlBuJ1Envl1KAhwBDzieeg_Z0v5ysDZcfuG6EXzhVCGMVIeZeAkxHgLi_bv63hA2JUeTPe1pZhUdWUnXIM0r7JmnMqs-lLr_G7OQOxO5sGCDmDhhEvHbYewQ7Zeq47gygklHfB7Ka0lkZXB-HaGZ3kyO0dBju8tZRC4YcgLRoHvFxmZCxvsGRocvzZYCpDMt2hkvvaJY9PigYzffYwBI50dCogBBZt7vKl4L42V_lBkRgMFno2hN1BLq52PWii89ecwS-mi4RkXuBEXuow_CThXB3ZKMCjAqtGnt6QcdSrpsxXLECWBtpdSpW5I4kP7tgP18N0MtHHSOqyt9V2WeoATjrxuQzqJO1Rbk9.NVpPv9SBy6H6lJsdQCcAWw/gweird_captions.json", "caps1": "https://www.kaggleusercontent.com/kf/343448140/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..5heDrYSMCkwgjEhje_gvpg.73IdaOvH0ullHD5bMeImFKwKzpwR9n12P-PAJ805kv1EKIeWfKLGBrP1Uj4Iqr2bdOOeZFBTDh0SrxC-JvO5G5ecWlRZetchy6i-K3pxxJzbBd2vErrAHo1lmiNplaWvQ5B8-Zcrc1fzrjH7KHLyrT5vpCAfSjxGCT7veXdhlK2-Xp87oF1u_wJ2V-0fziOTFdgnKWbVFxOsLC0wmHN3JdlMMrCFLdOba3GBSjZp-HIej81v1wqsDWbw2l52OLxOIk3eZ9RuClICxi5q2bT-PGJBnHBtHJ67OJ8M53Nbqg9310oj6w6oozuca1YP8RRAz7iMq7r_uesrkv90_JFuRjtKJ2MQee_5BmN1gZgP9ke0x-X0CnXV9E9avaOgdxoPtl72vw-TfwJzQnhJX0ws6RPqk-xrMgOQfH_Em94YJQh8Bgmkpf9OrX67gfYuksYtKFDFV5EckZ2SQCPLxu17NQ89BsPP0-62e2xYXp3bXC_Sp_CZS-io2Asoaph3ZOojilznJFVAA2dQ9t9u6-2w2peP_LlPa0iErC4mrmdkH88xUnh3WqUaS6LE5E87wGchir8QU88ySqZ7hNPFRcgQ9fqkdESjfILCDGYLcT152Gs5uYZXuF6xyFaiw_GC5-Rl.DrN9TQPCjIycES7oS8BLgw/gweird_captions.json", "caps2": "https://www.kaggleusercontent.com/kf/343562607/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..XS6dy6DuyXMNAKhB7TsPrw.dhvssFq4Ng4MXvcQkfcVTlo46WM4s0exCc02vv7eH-n-UqI-K_J3kb7JAFX8VerrOOsmoGdbRCG_M7H-sKT5y-UqXgF1G1O0aNtxXT_IaX3Z1_6Cb2vhrgiBKrYot7YLseUxheI_53YPyIbjWMmPKat-fuiRs6oU1UVSojKoPJ9wkf72wmS4YhAe7ACosI7ANvKAaJqIEkLm_xu98SBtdtKoZxH4QOyF6HHnGMWBUQCk8UG-_MXQXyVa-e4ASGFYquJ89pkhmH6YGKe1xyH68W6c16554jUtIXwLFi9wpGzgdiEfpgY90iJL2PYw5suDeM6Goeo7yFukuDemc1owmm76aGBwZbpXUz43_82tMrb_0Pn67H7iWnjKh3Ox1wc0AGPkmCIvEE4XlQfVs8w1Vqz4ztw0MlJcSCR9GCOgUQOHvqRzhE_Y3I7ZHnRXd-IKDXQnS6c-Os2XDEF3XbFyas00GcEAEaFYcm3YIjv4iv5dAnNI8vpByPYWrz7ih4PKcU18UmOxAmZoYqFrIi7t-ei-3bfR7jSxF1kX68DCwDUgOxUU7ncK4EPcjCxZfO5do10WiBDGGVjquoJNoqFLWH9ANl6Zp2bLTQCGtuwUmtQiGBm9H8or5ckc3lfwsk_B.gK_LUsOuMvW-kOVsVj8lQA/gweird_captions.json", "caps3": "https://www.kaggleusercontent.com/kf/343562612/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..57UmDZTvHz1XQ3ZGbG2Csg.dq7VCsfDZxrPwP03Tl_pXsuysqrM7-B8nwH2XIH25Ebqkni3Rjy_Au04KbsYUgYgdbp1Pza-sJ97KFRgwIh8TSe0mOGI9K9UrAfttCV6Kq3ZVUMD7LB0g0qO-CSWF3DqpY1MY2-s2DaSRN67aNMjRvAq2BNb-EK9s6uRkBNh2Bt8URwR9x1opAuFLhSuKivXgb61fi3AX5_cbqrut6iZrtrJ0hmpPvsP3cqDXzKau_j9vi1NuDazIT7TVSCv3o0LSbDuQTGHxBtE95TlhM5PpB1KvrljbC7px_OTlGj79TiK1widxwKgJH-_5SJwuFmPhBNR1mwfinOS4UoflGAna5qw6z-83uJam0oYUsJKzcAhapfHCDMtqxIEYe3glBZ2_rP8BIpprKrkN5wOw5w935-6lrmyHz_I7G3asi9nQNUar9o22OlFFIkKufcotp054eFtHsHNZRJmh1p4czWqQf7N2GyGYAWNi81PNoiEvOuwfB-yxaTz-N_Cy1aZvRraqcvrCb70uJDQ_RFk3bU65meDD9aZ19WJzZ1lShrW0pHnJSJRcOpdfHDgVpXM_CmctZN0ZuSwohPQkH0wFjNMxNvOs5LNOm2FP0QR2_-nDqxKWinHVReHDol-Oh6CwmRA.Lj-icpKSqNGWBYcpGHedxA/gweird_captions.json", "caps4": "https://www.kaggleusercontent.com/kf/343562621/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..I4y7RSq5FRQRLhUNkLStrQ.je8OYlAi3tLuCwv6RPz563slaofOAair94DmopXF2ZzfKT4tXTLwGaUthKsSTnW5H2TY7d0YYw6F56lazbSTZOUkd_NQ1pgIPL6x5tN9B8RU9M-Lrun1X9NnJggrp3oZhYB2x5DscpN1d3ckFQhTLb8RYMcQlZ6LKGTGJ988a2hvb6SLAh6UdRfKaXmxslftZoPH8LBoVutahLo9uBLoS2JfBaPxRGvpFaN-flbQNQNT2peZ2CMaJbas6RZ-t9WlYaXNmMx704DREctxTUaJXldsk5TTDZqjZRJUymkdELdhq6apmhmaC23WHwFNc0BUu8MfSF9gX0rJbBhb7IjcGXl07Wwa8r_vlfEZ6zkYtBVUQsuPYihj045TFy68aV--BGnn-TJLD2-PApAgN-9nGdQej7r6aAiUQOr9aiab7PiP-J4lFi5Ws8DiyQOlBL8YTAYV2QrAFlENmpjxDysyzrseszV72MLXaLJnkK7Pf3FuXnzARKOwz4Pumu5AU0Xh6Loo4OKC_j0DiKwWf-aVNeUU3lrqU-tmi_bvuhkRA5iMgeUtt8ojgNu8uDb2wf2r-KTnVMGA_-oAyHwu2qM1zLzDF6pjkiyaGfHJkt3ghncHrY1U7nSJ6VdDKgqJQt3N.JG1Qbvy2QqTx_66O0ki3Og/gweird_captions.json", "meta5": "https://www.kaggleusercontent.com/kf/343562626/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..0ow4kMqml26itHEuAVq73w.Q0r9k7wZijgjeBgxHv7RyYWcfTWVB7m4I53na9HImZ151lMFhmFf2jVhrKnpvi22dWqFRvIGQspmFw2u5uNsC9M0cHYxn-Xo3Pyo251iThe_uj_li0CnSGBgT22lGRoYH1-Qj2_GbkIcrYhnd8WQhOslKpymUUMBPMovYrYStES3R-qQjc3cuvVslY6dI46xwUivOa0usVWPgo9wdfmyI4bufqurB_qWi_oyiMBhkBsSKuDN_9lgozl-rLS5tffYLhlWjFCoLCpjfMIkdG8lS1_bq1ImsfPvaDKH54U1rjH3HY0lPwf1XkjIMNyip4VTlVNclkUZoqTc08k0zfVWj9YDT3hmSKwF6Rk_rk2Bthn-ovdlhfGHSUnvaDp5GHoAXJJ83qvIs3x66ClFWPdyebXy6u6sckgPS_OHc9V0vT8hf82chB1QiIh_v1Rj06nGkA2KVdrh7L70GAPVjQ3DKxnwMYFIHC4nSxYne7OkeRvRdqkBscR9P4A62r6v1Rdi1ODHafjE3OshUNykAFaWpHNPtywMlYep2NrId5LAW3pGd-uQDx90ypWS9KDPS8nnqUV3cBomTFOMvUP37OjiZWKVIm1M7BCcGOJXKRMWuv52l2YswvhH2wRY6j1KddiD.Uo0vZpjaptoQgtLQvpON3Q/gweird_meta.json", "offs5": "https://www.kaggleusercontent.com/kf/343562626/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..xbXQfL9vfuS5Oz3bwDTcAw.pAa2iS3maD3vB9uaKg1jDFtnWBWVb_7tDQcKal4p5JZUIng-w88EGbWzn7ioI1IN-T-KGT4LYEGdehaA4M-gk8Za1zS6WOoG-RjbXbuAp02EqmAcaRFG-5yyBr3mUMPhMwFaPya6Hr5qfVD0jVuL5hSHiGYQ3iKzs1awqmA1O8hgTS3oJtSv8AdgTxhGgBYDXDcao3Fu8aZuGDN0BmKgMeQYP35HB6SqVv4GUI3pZmljUGWC_w4yvWq--nBfBkSIbDbRHFMsnb2sN0ec4mAMs36tqpThOggzgZtFfme3pOU3Jh7JgtP3wKvnh6UHLq9l_41RtQwcNBKWWLIKHyxmgL0Klreb77FFKzz04tQJGcHhLPAIPS00rPEDXX02OYaFjIQVWtneuEXw2sOCqhkUI6UWuTvSn-ACou82pEv8FF_gh26pE86odEgqqXOD8BAkLh23405b-ee1d5KHTYN2SYVjR2tRloplkJWjaI8RLj1dlmBKD3A-Xb2zRvHGSudn1jLpSGkcUE-yF9S6-e4S8zyfIXJj6QM3HSQL5InaeITOe6O_WpuW3AjEGs8kbhXXBkDlV_zRSg9jmD_IF9dvGv5liiQDshAOK-Y9G5vo6yqmqhh3dv5riaED8oY_1MR8.0UBpzBaCmW26Ha6gLhdeOg/gweird_offsets.json", "caps5": "https://www.kaggleusercontent.com/kf/343562626/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..QCU4nKsyMprs7mV0Fvxa_A.mh_XYv8rEjRM_w7J60X3etjdMaevphtk9C8NSSvgKCaiV4nYtIMIBsHznjrH_zktYxf8b9YAbOeSrSWAhKghoAdV6kei4pka2szOJggqb9ux_072avJbHONul1_UJiCFupmH6VK7cBongU3lg1995DRVjiqtdZmPHYVwOrVqWHAauemUPkgiXAW-5xndwwQdEN1gCvaUJPsmVgfwOXbfM_uFZ34qgAa93fpR3jvCl96UN5LhLztQKxD-RyNU1XNinP5EXzJd37a21eXH2TzCdwxDzkY6azoIfl9R6D-XF_cvihVNsQ6S5fZFRx_Gf6JQ6tzpsc4sejorEnE3sW4wkrVu8lYPnuGDPPiaE9pc9GjyXGrkdLh9wl8WnkXL0eUV6IC4W20Os9iiCJiwpF18M70II_FmVokL0TesiB-UEivaNE2bitlDqP4mtKMUqkZmjF_IcYya4elEfI1n598ELxm7BailOL-SuoTr0noDwH2hOiz-tx02NT4940S1Fvp6od4htug9WKffjwlO62NS2gVhzpmMBQQsKLDcmKHoqtv1KS03DxzeF89IjwIkTiu22-dDQ9pgQ0jH523gNuXV70iVPoegvQ3BxwAnlmT22qvvl_Zt3epA06A6eywihWG4.vBvWOZerP0em6ZW7PfRCtw/gweird_captions.json", "img5": "https://www.kaggleusercontent.com/kf/343562626/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Z2zpRRfZ0GZJHaMeLSFu-A.pslrLIv_auflLDzUWwzEdh1fOY9A3GYLrvd7M6Prh2zHnkxNboQU79usyYz6Uh8k6hd2rZum6urEeZZfPJ7mdtydrG_whYDoYM_UvbuWyx0w0gqbooFnOX4X4di52g3hl4tf9eSLaSc-VjYHral4z9YOX8k_6PuBBimyRE021nDy9NN3Ds5Uv5_zl1dxgZqstueWSa4AzPYVFy8LgfQRcld_Hxrtu_cgXWTzp2bPc9eA_h0b-2FK3vAICZHkF57JyVYaCEF7xJDXOghB8EL3z-JG0fG1m0BvMb3fZJ4OVyZYzwghDjoo4myq08Hm_urMY-IBO1Wa0OTBOIMbmbse7s8rENyQH--2oLO2dYBGw-V0UQ-PUPgePSxTWFvvXfxQktXkmSuxDeG2a0EyFU29gsAcVqw17P-3y_My9hiG8eGdTwMLwspQQHEvd6Jy8W2nCURtKbkVjBZc8bUXYkAY47EKkMU1zy9IwDAigRylh13Y6mBCC1vCPtClkWCUDp0OusuK_tM-8PsOBwvoLqypJziNRI7x0KYn1qPVGeBIfg7xmjIr_bVD8Gkbx3q4mE4DaFtvlKTfZxduUack2Lt1nfLVVtiwUBbEQuzFZAEJZl5RzsTmlIboE9EwFfeiW3DA.p17myTuD_LYvpzWnq9AXmA/gweird_images.jpgbin", "vqvae": "https://www.kaggleusercontent.com/kf/344091927/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..na_qs1kvkSbXNcOrxGR-Ww.KGZa8WiojjD6NgTPjJde5PbA5pFCsXU7a_MYaV7AhrJhFt4xN7Lq44vgSB5tkdCoftrvVwCs1Jl7YABF5iLRVAXFtNJXtqHTiOTZHg7SzXDSSmHY-sjdr6m_ScIHSmW7mUuBV3gWU058_BbNM9BYNaW29i47xLjCnjj5TW81o_pfb8bbQS2OyZ8iWUgN6OgHgyUIRalBhXw8Z_vB4213VS0NctoZ8ihl3oLVYTMJU_RYzADFuCZssaR1Uxho-dlxCX29Q-KKk1xvGkGOr1w6fO0pX1IZAQvplKWiXSrHH1EB2O0z3CCJu7MkCt5GcAI5aLVHrWoa1hzaqjid-gKCnHVu5h6-Txws3ZEsTxi_lNO75df9aEG0hnzUck3I2J9MyqgPnTvYm0OtW8CXY8fJV2G4mdZgPFk7ggwHUeri8_tjzHL3iceW_NG-w1kFJAgbTi73-siOwxlBOp3rgE1pjcZpQxQKoJ-bOzbg5oYTo2Tvpg2OMVZod-y-JO8jeWsDHkUjs910L2eTHw8d5Kau7IJ5pLV46eYCzD1ZiCuWTDkfyiwfU50czsoKbqPVaonvc4XCe2xCH5M0TDh1w25bi9eLOG0YSF7F8sNDOd-floz3nqDAntpkfbjxzHdSC4-y.kbQ1zzus7TDmDVz1GWGmOA/run/vqvae.pt"}')

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
