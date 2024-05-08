from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def merge_2023(folder=Path("2023")):
    df_SRL = pd.read_excel(
        folder / "SRL.xlsx", sheet_name="Beregningsskjema 2023", skiprows=1
    )
    df_SRL = df_SRL[1:-7]

    df_SMET = pd.read_excel(
        folder / "SMET.xlsx", sheet_name="Beregningsskjema 2023", skiprows=1
    )
    df_SMET = df_SMET[1:-7]

    df_SC = pd.read_excel(
        folder / "SC.xlsx", sheet_name="Beregningsskjema 2023", skiprows=1
    )
    df_SC = df_SC[1:-7]

    df_SUIB = pd.read_excel(
        folder / "SUIB.xlsx", sheet_name="Beregningsskjema 2023", skiprows=1
    )
    df_SUIB = df_SUIB[1:-7]

    with pd.option_context("future.no_silent_downcasting", True):
        df_SRL = df_SRL.fillna(0)
        df_SMET = df_SMET.fillna(0)
        df_SC = df_SC.fillna(0)
        df_SUIB = df_SUIB.fillna(0)

    eks_år = df_SRL[" Eks. år     "]
    ant_pers = (
        df_SRL["Ant pers i bedrifts-gruppen"]
        + df_SMET["Ant pers i bedrifts-gruppen"]
        + df_SC["Ant pers i bedrifts-gruppen"]
        + df_SUIB["Ant pers i bedrifts-gruppen"]
    )
    lm = (
        df_SRL["lm (gruppens lønnsmasse)"]
        + df_SMET["lm (gruppens lønnsmasse)"]
        + df_SC["lm (gruppens lønnsmasse)"]
        + df_SUIB["lm (gruppens lønnsmasse)"]
    )
    gj_snitt = np.divide(
        lm.to_numpy(),
        ant_pers.to_numpy(),
        out=np.zeros_like(lm.to_numpy()),
        where=ant_pers.to_numpy() > 0,
    )
    arm = df_SRL[" Glattet aritmetisk middel for Tekna (X)"]

    armp1 = df_SRL["Glattet Aritmetisk middel for Tekna +1 eks.år (X-1)"]
    ak_tilegg = df_SRL["AK-tillegg"]
    ak_tillegg_gr = ant_pers * ak_tilegg

    df = pd.DataFrame(
        {
            " Eks. år     ": eks_år,
            "Ant pers i bedrifts-gruppen": ant_pers,
            "lm (gruppens lønnsmasse)": lm,
            "Gruppens gj.snitts-lønn": gj_snitt,
            " Glattet aritmetisk middel for Tekna (X)": arm,
            "Glattet Aritmetisk middel for Tekna +1 eks.år (X-1)": armp1,
            "AK-tillegg": ak_tilegg,
            "Gruppens AK-tillegg": ak_tillegg_gr,
        }
    )
    df.to_excel("2023.xlsx", index=False)

    lm_sum = lm.sum()
    LM_sum = (arm * ant_pers).sum()
    print("2023: 100 * lm / LM = ", 100 * lm_sum / LM_sum)

    ak_tillegg_gr_sum = ak_tillegg_gr.sum()
    print("Gjennomsnittlig AK-tillegg i 2023: ", 100 * ak_tillegg_gr_sum / LM_sum)


def plot_gjennomsnittslønn(df_2023, df_2022, df_2021, df_2020):
    # Plot bar chart with salary on y-axis and Eks. år on x-axis with different colors for each year
    fig, ax = plt.subplots(figsize=(12, 5))

    y_2023 = df_2023["Gruppens gj.snitts-lønn"].to_numpy()
    y_2022 = np.zeros_like(y_2023)
    y_2022[1:] = df_2022["Gruppens gj.snitts-lønn"].to_numpy()[:-1]
    y_2021 = np.zeros_like(y_2023)
    y_2021[2:] = df_2021["Gruppens gj.snitts-lønn"].to_numpy()[:-2]
    y_2020 = np.zeros_like(y_2023)
    y_2020[3:] = df_2020["Gruppens gj.snitts-lønn"].to_numpy()[:-3]

    x = np.arange(len(y_2023))
    width = 0.2
    ax.bar(
        x - 2 * width,
        y_2020,
        width=width,
        color="b",
        label="2020",
    )
    ax.bar(
        x - width,
        y_2021,
        width=width,
        color="r",
        label="2021",
    )
    ax.bar(
        x,
        y_2022,
        color="g",
        width=width,
        label="2022",
    )
    ax.bar(x + width, y_2023, color="y", width=width, label="2023")
    ax.set_xticks(x)
    ax.set_xticklabels(df_2023[" Eks. år     "], rotation=45)
    ax.set_xlabel("Eks. år")
    ax.set_ylabel("Gjennomsnittslønn MNOK")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig("gjennomsnittslønn.png")


def plot_differanse_tekna_aritmetisk_middel(df_2023, df_2022, df_2021, df_2020):
    fig, ax = plt.subplots(figsize=(12, 5))

    y_2023_lønn = np.nan_to_num(
        df_2023["Gruppens gj.snitts-lønn"].to_numpy().astype(float), 0.0
    )
    y_2023 = (
        y_2023_lønn - df_2023[" Glattet aritmetisk middel for Tekna (X)"].to_numpy()
    )
    y_2023[np.isclose(y_2023_lønn, 0.0)] = 0.0

    y_2022_lønn = np.nan_to_num(
        df_2022["Gruppens gj.snitts-lønn"].to_numpy().astype(float), 0.0
    )
    y_2022 = np.zeros_like(y_2023)
    y_2022[1:] = (
        y_2022_lønn[:-1]
        - df_2022[" Glattet aritmetisk middel for Tekna (X)"].to_numpy()[:-1]
    )
    y_2022[1:][np.isclose(y_2022_lønn[:-1], 0.0)] = 0.0

    y_2021_lønn = np.nan_to_num(
        df_2021["Gruppens gj.snitts-lønn"].to_numpy().astype(float), 0.0
    )
    y_2021 = np.zeros_like(y_2023)
    y_2021[2:] = (
        y_2021_lønn[:-2]
        - df_2021[" Glattet aritmetisk middel for Tekna (X)"].to_numpy()[:-2]
    )
    y_2021[2:][np.isclose(y_2021_lønn[:-2], 0.0)] = 0.0

    y_2020_lønn = np.nan_to_num(
        df_2020["Gruppens gj.snitts-lønn"].to_numpy().astype(float), 0.0
    )
    y_2020 = np.zeros_like(y_2023)
    y_2020[3:] = (
        y_2020_lønn[:-3]
        - df_2020[" Glattet aritmetisk middel for Tekna (X)"].to_numpy()[:-3]
    )
    y_2020[3:][np.isclose(y_2020_lønn[:-3], 0.0)] = 0.0

    x = np.arange(len(y_2023))
    width = 0.2
    ax.bar(
        x - 2 * width,
        y_2020,
        width=width,
        color="b",
        label="2020",
    )
    ax.bar(
        x - width,
        y_2021,
        width=width,
        color="r",
        label="2021",
    )
    ax.bar(
        x,
        y_2022,
        color="g",
        width=width,
        label="2022",
    )
    ax.bar(x + width, y_2023, color="y", width=width, label="2023")
    ax.set_xticks(x)
    ax.set_xticklabels(df_2023[" Eks. år     "], rotation=45)
    ax.set_xlabel("Eks. år")
    ax.set_ylabel("Differanse Tekna - Gruppens gjennomsnittslønn MNOK")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig("differanse_tekna_aritmetisk_middel.png")

    ax.set_ylim(-400_000, 200_000)
    fig.tight_layout()
    fig.savefig("differanse_tekna_aritmetisk_middel_ylim.png")

    diff_2023_prosent = 100 * np.nan_to_num(
        np.divide(
            y_2023,
            y_2023_lønn,
            out=np.zeros_like(y_2023),
            where=y_2023_lønn != 0,
        ),
        0.0,
    )
    personer_2023 = np.nan_to_num(
        df_2023["Ant pers i bedrifts-gruppen"].to_numpy().astype(float), 0.0
    )
    diff_2023_prosent_avg = (
        diff_2023_prosent * personer_2023
    ).sum() / personer_2023.sum()
    print("2023: Gjennomsnittlig prosentvis differanse: ", diff_2023_prosent_avg)

    diff_2022_prosent = 100 * np.nan_to_num(
        np.divide(
            y_2022[1:],
            y_2022_lønn[:-1],
            out=np.zeros_like(y_2022[1:]),
            where=y_2022_lønn[:-1] != 0,
        ),
        0.0,
    )
    personer_2022 = np.nan_to_num(
        df_2022["Ant pers i bedrifts-gruppen"].to_numpy().astype(float)[:-1], 0.0
    )
    diff_2022_prosent_avg = (
        diff_2022_prosent * personer_2022
    ).sum() / personer_2022.sum()
    print("2022: Gjennomsnittlig prosentvis differanse: ", diff_2022_prosent_avg)

    diff_2021_prosent = 100 * np.nan_to_num(
        np.divide(
            y_2021[2:],
            y_2021_lønn[:-2],
            out=np.zeros_like(y_2021[2:]),
            where=y_2021_lønn[:-2] != 0,
        ),
        0.0,
    )
    personer_2021 = np.nan_to_num(
        df_2021["Ant pers i bedrifts-gruppen"].to_numpy().astype(float)[:-2], 0.0
    )
    diff_2021_prosent_avg = (
        diff_2021_prosent * personer_2021
    ).sum() / personer_2021.sum()
    print("2021: Gjennomsnittlig prosentvis differanse: ", diff_2021_prosent_avg)

    diff_2020_prosent = 100 * np.nan_to_num(
        np.divide(
            y_2020[3:],
            y_2020_lønn[:-3],
            out=np.zeros_like(y_2020[3:]),
            where=y_2020_lønn[:-3] != 0,
        ),
        0.0,
    )
    personer_2020 = np.nan_to_num(
        df_2020["Ant pers i bedrifts-gruppen"].to_numpy().astype(float)[:-3], 0.0
    )
    diff_2020_prosent_avg = (
        diff_2020_prosent * personer_2020
    ).sum() / personer_2020.sum()
    print("2020: Gjennomsnittlig prosentvis differanse: ", diff_2020_prosent_avg)

    fig, ax = plt.subplots()
    x = [2020, 2021, 2022, 2023]
    y = [
        diff_2020_prosent_avg,
        diff_2021_prosent_avg,
        diff_2022_prosent_avg,
        diff_2023_prosent_avg,
    ]
    ax.plot(x, y, marker="o")
    ax.set_xlabel("År")
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.set_ylabel("Gjennomsnittlig prosentvis differanse fra Tekna")
    ax.grid()
    fig.tight_layout()
    fig.savefig("gjennomsnittlig_prosentvis_differanse.png")


def plot_gjennomsnitt_prosentvis_endring(df_2023, df_2022, df_2021, df_2020):
    fig, ax = plt.subplots(figsize=(12, 5))

    y_2023 = df_2023["Gruppens gj.snitts-lønn"].to_numpy()
    y_2022 = np.zeros_like(y_2023)
    y_2022[1:] = df_2022["Gruppens gj.snitts-lønn"].to_numpy()[:-1]
    y_2021 = np.zeros_like(y_2023)
    y_2021[2:] = df_2021["Gruppens gj.snitts-lønn"].to_numpy()[:-2]
    y_2020 = np.zeros_like(y_2023)
    y_2020[3:] = df_2020["Gruppens gj.snitts-lønn"].to_numpy()[:-3]

    y_2023_prosent = 100 * np.nan_to_num(
        np.divide(
            y_2023 - y_2022, y_2022, out=np.zeros_like(y_2023), where=y_2022 != 0
        ),
        0.0,
    )
    y_2022_prosent = 100 * np.nan_to_num(
        np.divide(
            y_2022 - y_2021, y_2021, out=np.zeros_like(y_2022), where=y_2021 != 0
        ),
        0.0,
    )
    y_2021_prosent = 100 * np.nan_to_num(
        np.divide(
            y_2021 - y_2020, y_2020, out=np.zeros_like(y_2021), where=y_2020 != 0
        ),
        0.0,
    )

    # Legg til forhandlet prosentvis endring
    y_2023_prosent = np.append(5.2, y_2023_prosent)
    y_2022_prosent = np.append(4.38, y_2022_prosent)
    y_2021_prosent = np.append(3.3, y_2021_prosent)

    x = np.arange(len(y_2023_prosent))
    width = 0.2
    ax.bar(
        x - width,
        y_2021_prosent,
        width=width,
        color="b",
        label="2021",
    )
    ax.bar(
        x,
        y_2022_prosent,
        width=width,
        color="r",
        label="2022",
    )
    ax.bar(
        x + width,
        y_2023_prosent,
        color="g",
        width=width,
        label="2023",
    )
    ax.set_xticks(x)
    labels = ["Forhandlet"] + df_2023[" Eks. år     "].to_list()
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel("Eks. år")
    ax.set_ylabel("Historisk prosentvis endring")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig("gjennomsnitt_prosentvis_endring.png", dpi=300)


def forventet_ak_tillegg_med_kpi(df_2023, df_2022, df_2021, df_2020):
    kpi_2024 = 5.5
    kpi_2023 = 5.5
    kpi_2022 = 6.5
    kpi_2021 = 5.4

    y_2023 = df_2023["Gruppens gj.snitts-lønn"].to_numpy()
    y_2022 = np.zeros_like(y_2023)
    y_2022[1:] = df_2022["Gruppens gj.snitts-lønn"].to_numpy()[:-1]
    y_2021 = np.zeros_like(y_2023)
    y_2021[2:] = df_2021["Gruppens gj.snitts-lønn"].to_numpy()[:-2]
    y_2020 = np.zeros_like(y_2023)
    y_2020[3:] = df_2020["Gruppens gj.snitts-lønn"].to_numpy()[:-3]

    y_2023_ak_med_kpi = y_2023 * (1 + kpi_2024 / 100) + df_2023["Gruppens AK-tillegg"]
    y_2022_ak_med_kpi = np.zeros_like(y_2023_ak_med_kpi)
    y_2022_ak_med_kpi[1:] = (
        y_2022[1:] * (1 + kpi_2023 / 100) + df_2022["Gruppens AK-tillegg"][:-1]
    )
    y_2021_ak_med_kpi = np.zeros_like(y_2023_ak_med_kpi)
    y_2021_ak_med_kpi[2:] = (
        y_2021[2:] * (1 + kpi_2022 / 100) + df_2021["Gruppens AK-tillegg"][:-2]
    )
    y_2020_ak_med_kpi = np.zeros_like(y_2023_ak_med_kpi)
    y_2020_ak_med_kpi[3:] = (
        y_2020[3:] * (1 + kpi_2021 / 100) + df_2020["Gruppens AK-tillegg"][:-3]
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(y_2023))
    width = 0.2
    ax.bar(
        x - 2 * width,
        y_2020_ak_med_kpi,
        width=width,
        color="b",
        label="2020",
    )
    ax.bar(
        x - width,
        y_2021_ak_med_kpi,
        width=width,
        color="r",
        label="2021",
    )
    ax.bar(
        x,
        y_2022_ak_med_kpi,
        color="g",
        width=width,
        label="2022",
    )
    ax.bar(x + width, y_2023_ak_med_kpi, color="y", width=width, label="2023")
    ax.set_xticks(x)
    ax.set_xticklabels(df_2023[" Eks. år     "], rotation=45)
    ax.set_xlabel("Eks. år")
    ax.set_ylabel("Forventet lønn Tekna (AK-tillegg med KPI) MNOK")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig("forventet_ak_tillegg_med_kpi.png")

    y_2023_ak_med_kpi_endring = y_2023_ak_med_kpi - y_2023
    y_2022_ak_med_kpi_endring = y_2022_ak_med_kpi - y_2022
    y_2021_ak_med_kpi_endring = y_2021_ak_med_kpi - y_2021
    y_2020_ak_med_kpi_endring = y_2020_ak_med_kpi - y_2020

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(y_2023))
    width = 0.2
    ax.bar(
        x - 2 * width,
        y_2020_ak_med_kpi_endring,
        width=width,
        color="b",
        label="2020",
    )
    ax.bar(
        x - width,
        y_2021_ak_med_kpi_endring,
        width=width,
        color="r",
        label="2021",
    )
    ax.bar(
        x,
        y_2022_ak_med_kpi_endring,
        color="g",
        width=width,
        label="2022",
    )
    ax.bar(x + width, y_2023_ak_med_kpi_endring, color="y", width=width, label="2023")
    ax.set_xticks(x)
    ax.set_xticklabels(df_2023[" Eks. år     "], rotation=45)
    ax.set_xlabel("Eks. år")
    ax.set_ylabel("Forventet endring med AK-tillegg og KPI i MNOK")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig("forventet_ak_tillegg_med_kpi_endring.png")

    y_2023_ak_med_kpi_endring_prosent = 100 * np.nan_to_num(
        np.divide(
            y_2023_ak_med_kpi_endring,
            y_2023,
            out=np.zeros_like(y_2023),
            where=y_2023 != 0,
        ),
        0.0,
    )
    y_2022_ak_med_kpi_endring_prosent = 100 * np.nan_to_num(
        np.divide(
            y_2022_ak_med_kpi_endring,
            y_2022,
            out=np.zeros_like(y_2022),
            where=y_2022 != 0,
        ),
        0.0,
    )
    y_2021_ak_med_kpi_endring_prosent = 100 * np.nan_to_num(
        np.divide(
            y_2021_ak_med_kpi_endring,
            y_2021,
            out=np.zeros_like(y_2021),
            where=y_2021 != 0,
        ),
        0.0,
    )
    y_2020_ak_med_kpi_endring_prosent = 100 * np.nan_to_num(
        np.divide(
            y_2020_ak_med_kpi_endring,
            y_2020,
            out=np.zeros_like(y_2020),
            where=y_2020 != 0,
        ),
        0.0,
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(y_2023))
    width = 0.2
    ax.bar(
        x - 2 * width,
        y_2020_ak_med_kpi_endring_prosent,
        width=width,
        color="b",
        label="2020",
    )
    ax.bar(
        x - width,
        y_2021_ak_med_kpi_endring_prosent,
        width=width,
        color="r",
        label="2021",
    )
    ax.bar(
        x,
        y_2022_ak_med_kpi_endring_prosent,
        color="g",
        width=width,
        label="2022",
    )
    ax.bar(
        x + width,
        y_2023_ak_med_kpi_endring_prosent,
        color="y",
        width=width,
        label="2023",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df_2023[" Eks. år     "], rotation=45)
    ax.set_xlabel("Eks. år")
    ax.set_ylabel("Forventet endring med AK-tillegg og KPI i prosent")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig("forventet_ak_tillegg_med_kpi_endring_prosent.png")


def plot_lønnsvekts():
    år = [2020, 2021, 2022, 2023]
    kpi = [5.4, 6.5, 5.5, 4.0]
    lønnsvekst = [3.3, 4.38, 5.2]

    fig, ax = plt.subplots()
    ax.plot(år, kpi, marker="o", label="KPI")
    ax.plot(år[:-1], lønnsvekst, marker="o", label="Lønnsvekst")
    ax.set_xlabel("År")
    ax.set_xticks(år)
    ax.set_xticklabels(år)
    ax.set_ylabel("Prosent")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig("lønnsvekst.png")


def plot_ak_tillegg(df_2023):
    fig, ax = plt.subplots()
    x = df_2023[" Eks. år     "]
    y = df_2023["AK-tillegg"]
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Eksamens år")
    ax.set_ylabel("AK-tillegg")
    ax.grid()
    fig.tight_layout()
    fig.savefig("ak_tillegg.png")


def main():
    merge_2023()
    df_2023 = pd.read_excel("2023.xlsx")

    df_2020 = pd.read_excel("2020.xlsx", sheet_name="Beregningssjema", skiprows=1)
    df_2020 = df_2020[1:-7]

    df_2021 = pd.read_excel("2021.xlsx", sheet_name="Beregningssjema", skiprows=1)
    df_2021 = df_2021[1:-7]

    df_2022 = pd.read_excel("2022.xlsx", sheet_name="Beregningssjema", skiprows=1)
    df_2022 = df_2022[1:-7]

    plot_gjennomsnittslønn(df_2023, df_2022, df_2021, df_2020)
    plot_differanse_tekna_aritmetisk_middel(df_2023, df_2022, df_2021, df_2020)
    plot_gjennomsnitt_prosentvis_endring(df_2023, df_2022, df_2021, df_2020)
    forventet_ak_tillegg_med_kpi(df_2023, df_2022, df_2021, df_2020)
    plot_lønnsvekts()
    plot_ak_tillegg(df_2023)


if __name__ == "__main__":
    main()
