import pandas as pd
import numpy as np

# Create the data manually
data = """YEAR_MONTH,YEAR,MONTH,ISSUER,SPEND,TRANSACTIONS,CARDS
201905,2019,5,Client,1246967699,25415401,1223816
201905,2019,5,Market,4243021441,93572826,4117556
201906,2019,6,Client,1235888088,25457565,1226961
201906,2019,6,Market,4048216335,91333643,4114516
201907,2019,7,Client,1276158437,26340731,1231457
201907,2019,7,Market,4179180495,95418264,4147717
201908,2019,8,Client,1280466654,26371664,1241890
201908,2019,8,Market,4009081823,93486730,4141316
201909,2019,9,Client,1264778527,25875957,1137221
201909,2019,9,Market,4079149641,93212672,4149831
201910,2019,10,Client,1281230919,27637228,1144521
201910,2019,10,Market,4148290208,97960499,3690864
201911,2019,11,Client,1332576396,27901198,1155441
201911,2019,11,Market,4039913286,98359899,4184421
201912,2019,12,Client,1509200910,31065923,1164513
201912,2019,12,Market,4272282206,106160685,4199818
202001,2020,1,Client,1244417250,26000948,1150502
202001,2020,1,Market,4033668388,91645166,4178060
202002,2020,2,Client,1149980384,25673854,1148920
202002,2020,2,Market,3644918169,89299470,4136820
202003,2020,3,Client,1305049948,29092605,1275667
202003,2020,3,Market,4365558411,102739358,4210453
202004,2020,4,Client,1296579998,29223503,1278435
202004,2020,4,Market,4259010906,99345440,4192805
202005,2020,5,Client,1328336725,30011032,1313950
202005,2020,5,Market,4204409387,103663927,4213943
202006,2020,6,Client,1337474194,29492525,1288624
202006,2020,6,Market,4026595624,101343716,4209559
202007,2020,7,Client,1388156418,30836904,1309431
202007,2020,7,Market,4158132595,105171400,4238899
202008,2020,8,Client,1405915522,30929984,1298832
202008,2020,8,Market,4009132218,104392402,4232928
202009,2020,9,Client,1333789907,29918064,1295033
202009,2020,9,Market,3959588585,101803688,4236040
202010,2020,10,Client,1379523108,31582579,1308935
202010,2020,10,Market,4206895913,108015300,4280120
202011,2020,11,Client,1439097698,31820523,1308599
202011,2020,11,Market,4209438511,107764556,4273156
202012,2020,12,Client,1668871938,35290486,1322951
202012,2020,12,Market,4997231298,123669354,4611630
202101,2021,1,Client,1363133304,29442093,1305829
202101,2021,1,Market,4304099828,100365946,4273078
202102,2021,2,Client,1297110418,29905000,1308311
202102,2021,2,Market,4183839054,104433523,4395189
202103,2021,3,Client,1289643764,28512477,1311634
202103,2021,3,Market,4214351678,100826641,4410232
202104,2021,4,Client,1040095706,22523868,1284351
202104,2021,4,Market,3361432690,68155886,4137136
202105,2021,5,Client,1280602508,27896183,1384700
202105,2021,5,Market,3959754131,92747440,4399784
202106,2021,6,Client,1333383855,31453374,1409200
202106,2021,6,Market,4786384021,103015058,4900678
202107,2021,7,Client,1527050863,35800980,1434733
202107,2021,7,Market,5569159494,121992035,5044838
202108,2021,8,Client,1587459003,36647094,1438394
202108,2021,8,Market,5963607193,131262894,5078154
202109,2021,9,Client,1539996674,34974972,1439462
202109,2021,9,Market,5493546178,126413369,5085434
202110,2021,10,Client,1554690117,35073728,1443019
202110,2021,10,Market,5449578162,129096281,5077628
202111,2021,11,Client,1355206039,33469174,1437589
202111,2021,11,Market,5236365674,117762121,5017104
202112,2021,12,Client,1764640776,40543804,1473788
202112,2021,12,Market,7071818233,155071263,5947303
202201,2022,1,Client,1266967043,29268054,1432717
202201,2022,1,Market,4304099828,100365946,4273078
202202,2022,2,Client,1216349483,29493453,1425992
202202,2022,2,Market,4875704948,107090382,4827375
202203,2022,3,Client,1381453793,35422145,1454870
202203,2022,3,Market,6103287748,126783411,5341773
202204,2022,4,Client,1380691955,35430970,1445894
202204,2022,4,Market,6373111105,132714362,5364241"""

from io import StringIO
df = pd.read_csv(StringIO(data))

# Calculate key metrics
df['SPEND_PER_CARD'] = df['SPEND'] / df['CARDS']
df['TXN_PER_CARD'] = df['TRANSACTIONS'] / df['CARDS']
df['AVG_TXN_VALUE'] = df['SPEND'] / df['TRANSACTIONS']

# Separate client and market data
client_df = df[df['ISSUER'] == 'Client'].copy()
market_df = df[df['ISSUER'] == 'Market'].copy()

# Calculate market share
market_share_df = pd.DataFrame()
market_share_df['YEAR_MONTH'] = client_df['YEAR_MONTH'].values
market_share_df['SPEND_SHARE'] = (client_df['SPEND'].values / (client_df['SPEND'].values + market_df['SPEND'].values)) * 100
market_share_df['TXN_SHARE'] = (client_df['TRANSACTIONS'].values / (client_df['TRANSACTIONS'].values + market_df['TRANSACTIONS'].values)) * 100
market_share_df['CARD_SHARE'] = (client_df['CARDS'].values / (client_df['CARDS'].values + market_df['CARDS'].values)) * 100

# Add date column for easier analysis
client_df['DATE'] = pd.to_datetime(client_df['YEAR_MONTH'], format='%Y%m')
market_df['DATE'] = pd.to_datetime(market_df['YEAR_MONTH'], format='%Y%m')
market_share_df['DATE'] = pd.to_datetime(market_share_df['YEAR_MONTH'], format='%Y%m')

# Define Repeel launch date (end of 2021)
repeel_launch = pd.to_datetime('2021-12-01')

# Create period flag
client_df['PERIOD'] = client_df['DATE'].apply(lambda x: 'Pre-Repeel' if x < repeel_launch else 'Post-Repeel')
market_df['PERIOD'] = market_df['DATE'].apply(lambda x: 'Pre-Repeel' if x < repeel_launch else 'Post-Repeel')

print("=" * 80)
print("KEY METRICS SUMMARY")
print("=" * 80)

# Pre-Repeel averages (excluding 2021 Dec which is launch month)
pre_client = client_df[client_df['PERIOD'] == 'Pre-Repeel']
post_client = client_df[client_df['PERIOD'] == 'Post-Repeel']
post_client_excl_dec = post_client[post_client['YEAR_MONTH'] != 202112]

pre_market = market_df[market_df['PERIOD'] == 'Pre-Repeel']
post_market = market_df[market_df['PERIOD'] == 'Post-Repeel']
post_market_excl_dec = post_market[post_market['YEAR_MONTH'] != 202112]

print("\nCLIENT (EUB) PERFORMANCE:")
print("-" * 80)
print(f"Pre-Repeel Average (May 2019 - Nov 2021):")
print(f"  Spend per Card: £{pre_client['SPEND_PER_CARD'].mean():,.0f}")
print(f"  Txn per Card: {pre_client['TXN_PER_CARD'].mean():.1f}")
print(f"  Avg Txn Value: £{pre_client['AVG_TXN_VALUE'].mean():.2f}")

print(f"\nPost-Repeel Average (Jan 2022 - Apr 2022):")
print(f"  Spend per Card: £{post_client_excl_dec['SPEND_PER_CARD'].mean():,.0f}")
print(f"  Txn per Card: {post_client_excl_dec['TXN_PER_CARD'].mean():.1f}")
print(f"  Avg Txn Value: £{post_client_excl_dec['AVG_TXN_VALUE'].mean():.2f}")

print(f"\nChange:")
spend_change = ((post_client_excl_dec['SPEND_PER_CARD'].mean() / pre_client['SPEND_PER_CARD'].mean()) - 1) * 100
txn_change = ((post_client_excl_dec['TXN_PER_CARD'].mean() / pre_client['TXN_PER_CARD'].mean()) - 1) * 100
avg_txn_change = ((post_client_excl_dec['AVG_TXN_VALUE'].mean() / pre_client['AVG_TXN_VALUE'].mean()) - 1) * 100

print(f"  Spend per Card: {spend_change:+.1f}%")
print(f"  Txn per Card: {txn_change:+.1f}%")
print(f"  Avg Txn Value: {avg_txn_change:+.1f}%")

print("\n" + "=" * 80)
print("MARKET PERFORMANCE:")
print("-" * 80)
print(f"Pre-Repeel Average:")
print(f"  Spend per Card: £{pre_market['SPEND_PER_CARD'].mean():,.0f}")
print(f"  Txn per Card: {pre_market['TXN_PER_CARD'].mean():.1f}")
print(f"  Avg Txn Value: £{pre_market['AVG_TXN_VALUE'].mean():.2f}")

print(f"\nPost-Repeel Average:")
print(f"  Spend per Card: £{post_market_excl_dec['SPEND_PER_CARD'].mean():,.0f}")
print(f"  Txn per Card: {post_market_excl_dec['TXN_PER_CARD'].mean():.1f}")
print(f"  Avg Txn Value: £{post_market_excl_dec['AVG_TXN_VALUE'].mean():.2f}")

market_spend_change = ((post_market_excl_dec['SPEND_PER_CARD'].mean() / pre_market['SPEND_PER_CARD'].mean()) - 1) * 100
market_txn_change = ((post_market_excl_dec['TXN_PER_CARD'].mean() / pre_market['TXN_PER_CARD'].mean()) - 1) * 100
market_avg_txn_change = ((post_market_excl_dec['AVG_TXN_VALUE'].mean() / pre_market['AVG_TXN_VALUE'].mean()) - 1) * 100

print(f"\nChange:")
print(f"  Spend per Card: {market_spend_change:+.1f}%")
print(f"  Txn per Card: {market_txn_change:+.1f}%")
print(f"  Avg Txn Value: {market_avg_txn_change:+.1f}%")

print("\n" + "=" * 80)
print("MARKET SHARE ANALYSIS:")
print("-" * 80)
pre_share = market_share_df[market_share_df['DATE'] < repeel_launch]
post_share = market_share_df[market_share_df['DATE'] >= repeel_launch]
post_share_excl_dec = post_share[post_share['YEAR_MONTH'] != 202112]

print(f"Pre-Repeel Average Market Share:")
print(f"  Spend Share: {pre_share['SPEND_SHARE'].mean():.2f}%")
print(f"  Txn Share: {pre_share['TXN_SHARE'].mean():.2f}%")
print(f"  Card Share: {pre_share['CARD_SHARE'].mean():.2f}%")

print(f"\nPost-Repeel Average Market Share:")
print(f"  Spend Share: {post_share_excl_dec['SPEND_SHARE'].mean():.2f}%")
print(f"  Txn Share: {post_share_excl_dec['TXN_SHARE'].mean():.2f}%")
print(f"  Card Share: {post_share_excl_dec['CARD_SHARE'].mean():.2f}%")

share_spend_change = post_share_excl_dec['SPEND_SHARE'].mean() - pre_share['SPEND_SHARE'].mean()
share_txn_change = post_share_excl_dec['TXN_SHARE'].mean() - pre_share['TXN_SHARE'].mean()
share_card_change = post_share_excl_dec['CARD_SHARE'].mean() - pre_share['CARD_SHARE'].mean()

print(f"\nChange in Market Share:")
print(f"  Spend Share: {share_spend_change:+.2f} pp")
print(f"  Txn Share: {share_txn_change:+.2f} pp")
print(f"  Card Share: {share_card_change:+.2f} pp")

print("\n" + "=" * 80)
print("DETAILED MONTH-OVER-MONTH TRENDS (Last 12 Months)")
print("=" * 80)

recent_client = client_df.tail(12)[['YEAR_MONTH', 'SPEND_PER_CARD', 'TXN_PER_CARD', 'AVG_TXN_VALUE']]
recent_market = market_df.tail(12)[['YEAR_MONTH', 'SPEND_PER_CARD', 'TXN_PER_CARD', 'AVG_TXN_VALUE']]
recent_share = market_share_df.tail(12)[['YEAR_MONTH', 'SPEND_SHARE', 'TXN_SHARE', 'CARD_SHARE']]

print("\nCLIENT:")
print(recent_client.to_string(index=False))

print("\n" + "=" * 80)

# Save data for visualization
client_df.to_csv('/home/claude/client_data.csv', index=False)
market_df.to_csv('/home/claude/market_data.csv', index=False)
market_share_df.to_csv('/home/claude/market_share.csv', index=False)

print("\n✓ Data saved for further analysis")

