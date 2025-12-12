
import streamlit as st
import pandas as pd
import sqlite3


st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    """
    Loads data from the SQLite database into two pandas DataFrames.
    The function is cached to improve performance.
    """
    # Connect to the database, assuming it's in a 'data' subdirectory
    conn = sqlite3.connect('data/housing_data.db') # MODIFIED PATH HERE
    df_eda = pd.read_sql_query("SELECT * FROM eda_data", conn)
    df_ml = pd.read_sql_query("SELECT * FROM ml_data", conn)
    conn.close()
    return df_eda, df_ml

# Example of how to use the loaded data in a Streamlit app
# if __name__ == "__main__":
#     st.title("Housing Price Analysis App")
#     df_eda, df_ml = load_data()

#     st.subheader("EDA Data Overview")
#     st.write(df_eda.head())

#     st.subheader("ML Data Overview")
#     st.write(df_ml.head())

#     st.success("Data loaded successfully from SQLite database!")

def eda_page():
    st.title('EDA Page')
    st.write('This is the EDA page.')

def prediction_page():
    st.title('Prediction Model (Coming Soon)')
    st.write('This page will contain the prediction model.')


def main():
    st.sidebar.title('Navigation')
    selected_page = st.sidebar.radio('Go to', ['EDA', 'Prediction Model'])

    if selected_page == 'EDA':
        eda_page()
    elif selected_page == 'Prediction Model':
        prediction_page()

# Run the app
if __name__ == '__main__':
    main()

def eda_page():
    st.title('Exploratory Data Analysis')
    
    df_eda, df_ml = load_data()

    st.header('Price & Size Analysis')

    # Q1 (Price Distribution)
    st.subheader('Distribution of Prices (Price_in_Lakhs)')
    fig_price_dist, ax_price_dist = plt.subplots(figsize=(10, 6))
    sns.histplot(df_ml['Price_in_Lakhs'], kde=True, ax=ax_price_dist)
    ax_price_dist.set_title('Distribution of Price in Lakhs')
    ax_price_dist.set_xlabel('Price in Lakhs')
    ax_price_dist.set_ylabel('Frequency')
    st.pyplot(fig_price_dist)

    # Q2 (Property Size)
    st.subheader('Distribution of Property Size (Size_in_SqFt)')
    fig_size_dist, ax_size_dist = plt.subplots(figsize=(10, 6))
    sns.histplot(df_ml['Size_in_SqFt'], kde=True, ax=ax_size_dist)
    ax_size_dist.set_title('Distribution of Size in SqFt')
    ax_size_dist.set_xlabel('Size in SqFt')
    ax_size_dist.set_ylabel('Frequency')
    st.pyplot(fig_size_dist)

    # Q3 (Price per SqFt by Property Type)
    st.subheader('Price per Square Foot by Property Type')
    fig_price_sqft_ptype, ax_price_sqft_ptype = plt.subplots(figsize=(12, 7))
    sns.boxplot(x='Property_Type', y='Price_per_SqFt', data=df_eda, ax=ax_price_sqft_ptype)
    ax_price_sqft_ptype.set_title('Price per SqFt Distribution by Property Type')
    ax_price_sqft_ptype.set_xlabel('Property Type')
    ax_price_sqft_ptype.set_ylabel('Price per SqFt')
    ax_price_sqft_ptype.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_price_sqft_ptype)

    # Q4 (Relationship between Size and Price)
    st.subheader('Relationship between Size in SqFt and Price in Lakhs')
    fig_size_price_rel, ax_size_price_rel = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Size_in_SqFt', y='Price_in_Lakhs', data=df_ml, ax=ax_size_price_rel)
    ax_size_price_rel.set_title('Relationship between Size in SqFt and Price in Lakhs')
    ax_size_price_rel.set_xlabel('Size in SqFt')
    ax_size_price_rel.set_ylabel('Price in Lakhs')
    ax_size_price_rel.grid(True)
    st.pyplot(fig_size_price_rel)

    correlation_coefficient = df_ml['Size_in_SqFt'].corr(df_ml['Price_in_Lakhs'])
    st.write(f"Pearson correlation coefficient between Size_in_SqFt and Price_in_Lakhs: {correlation_coefficient:.2f}")

    # Q5 (Outliers in Price per SqFt and Property Size)
    st.subheader('Outlier Detection in Price per SqFt and Size in SqFt')
    
    fig_price_sqft_outlier, ax_price_sqft_outlier = plt.subplots(figsize=(8, 6))
    sns.boxplot(y=df_ml['Price_per_SqFt'], ax=ax_price_sqft_outlier)
    ax_price_sqft_outlier.set_title('Box Plot of Price per SqFt')
    ax_price_sqft_outlier.set_ylabel('Price per SqFt')
    st.pyplot(fig_price_sqft_outlier)

    fig_size_outlier, ax_size_outlier = plt.subplots(figsize=(8, 6))
    sns.boxplot(y=df_ml['Size_in_SqFt'], ax=ax_size_outlier)
    ax_size_outlier.set_title('Box Plot of Size in SqFt')
    ax_size_outlier.set_ylabel('Size in SqFt')
    st.pyplot(fig_size_outlier)

    st.header('Numerical Features Analysis')

    # Q6 (BHK Distribution)
    st.subheader('Distribution of Number of Bedrooms (BHK)')
    fig_bhk_dist, ax_bhk_dist = plt.subplots(figsize=(10, 6))
    sns.histplot(df_ml['BHK'], kde=True, ax=ax_bhk_dist)
    ax_bhk_dist.set_title('Distribution of BHK')
    ax_bhk_dist.set_xlabel('BHK')
    ax_bhk_dist.set_ylabel('Frequency')
    st.pyplot(fig_bhk_dist)

    # Q7 (Property Age Distribution)
    st.subheader('Distribution of Property Age (Age_of_Property)')
    fig_age_dist, ax_age_dist = plt.subplots(figsize=(10, 6))
    sns.histplot(df_ml['Age_of_Property'], kde=True, ax=ax_age_dist)
    ax_age_dist.set_title('Distribution of Age of Property')
    ax_age_dist.set_xlabel('Age of Property')
    ax_age_dist.set_ylabel('Frequency')
    st.pyplot(fig_age_dist)

    # Q8 (Floor Numbers Distribution and Inconsistencies)
    st.subheader('Floor Numbers Distribution and Inconsistencies')
    fig_floor_no_dist, ax_floor_no_dist = plt.subplots(figsize=(10, 6))
    sns.histplot(df_ml['Floor_No'], kde=True, ax=ax_floor_no_dist)
    ax_floor_no_dist.set_title('Distribution of Floor Number')
    ax_floor_no_dist.set_xlabel('Floor Number')
    ax_floor_no_dist.set_ylabel('Frequency')
    st.pyplot(fig_floor_no_dist)

    st.write(f"Minimum Floor_No: {df_ml['Floor_No'].min()}")
    st.write(f"Maximum Floor_No: {df_ml['Floor_No'].max()}")
    st.write(f"Average Floor_No: {df_ml['Floor_No'].mean():.2f}")

    st.write(f"Minimum Total_Floors: {df_ml['Total_Floors'].min()}")
    st.write(f"Maximum Total_Floors: {df_ml['Total_Floors'].max()}")
    st.write(f"Average Total_Floors: {df_ml['Total_Floors'].mean():.2f}")

    # Verify if inconsistencies are resolved (Floor_No > Total_Floors)
    inconsistent_floors_after_fix = df_ml[df_ml['Floor_No'] > df_ml['Total_Floors']]
    if not inconsistent_floors_after_fix.empty:
        st.warning(f"Number of inconsistencies (Floor_No > Total_Floors): {len(inconsistent_floors_after_fix)}")
    else:
        st.success("No inconsistencies found where Floor_No is greater than Total_Floors.")

    # Q9 (Nearby Schools and Hospitals Counts)
    st.subheader('Amenities Count: Nearby Schools and Hospitals')

    fig_schools_hospitals, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df_ml['Nearby_Schools'], kde=True, bins=range(1, 12), ax=axes[0])
    axes[0].set_title('Distribution of Nearby_Schools')
    axes[0].set_xlabel('Number of Nearby Schools')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xticks(range(1, 11))

    sns.histplot(df_ml['Nearby_Hospitals'], kde=True, bins=range(1, 12), ax=axes[1])
    axes[1].set_title('Distribution of Nearby_Hospitals')
    axes[1].set_xlabel('Number of Nearby Hospitals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xticks(range(1, 11))
    plt.tight_layout()
    st.pyplot(fig_schools_hospitals)

    st.write('Descriptive Statistics for Nearby_Schools:')
    st.write(df_ml['Nearby_Schools'].describe())

    st.write('\nDescriptive Statistics for Nearby_Hospitals:')
    st.write(df_ml['Nearby_Hospitals'].describe())

    st.header('Categorical Features Analysis')

    # Q10 (Geographical Distribution - State)
    st.subheader('Geographical Distribution: States')
    state_counts = df_eda['State'].value_counts()
    st.write("Top 10 States by Number of Properties:")
    st.write(state_counts.head(10))
    fig_state_dist, ax_state_dist = plt.subplots(figsize=(12, 6))
    sns.barplot(x=state_counts.head(10).index, y=state_counts.head(10).values, ax=ax_state_dist)
    ax_state_dist.set_title('Top 10 States by Number of Properties')
    ax_state_dist.set_xlabel('State')
    ax_state_dist.set_ylabel('Number of Properties')
    ax_state_dist.set_xticklabels(ax_state_dist.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_state_dist)

    # Q10 (Geographical Distribution - City)
    st.subheader('Geographical Distribution: Cities')
    city_counts = df_eda['City'].value_counts()
    st.write("Top 10 Cities by Number of Properties:")
    st.write(city_counts.head(10))
    fig_city_dist, ax_city_dist = plt.subplots(figsize=(12, 6))
    sns.barplot(x=city_counts.head(10).index, y=city_counts.head(10).values, ax=ax_city_dist)
    ax_city_dist.set_title('Top 10 Cities by Number of Properties')
    ax_city_dist.set_xlabel('City')
    ax_city_dist.set_ylabel('Number of Properties')
    ax_city_dist.set_xticklabels(ax_city_dist.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_city_dist)

    # Q11 (Property Types)
    st.subheader('Distribution of Property Types')
    property_type_counts = df_eda['Property_Type'].value_counts()
    st.write("Distribution of Property Types:")
    st.write(property_type_counts)
    fig_prop_type_dist, ax_prop_type_dist = plt.subplots(figsize=(10, 6))
    sns.barplot(x=property_type_counts.index, y=property_type_counts.values, ax=ax_prop_type_dist)
    ax_prop_type_dist.set_title('Distribution of Property Types')
    ax_prop_type_dist.set_xlabel('Property Type')
    ax_prop_type_dist.set_ylabel('Number of Properties')
    ax_prop_type_dist.set_xticklabels(ax_prop_type_dist.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_prop_type_dist)

    # Q12 (Furnishing Status)
    st.subheader('Distribution of Furnishing Status')
    furnishing_status_counts = df_eda['Furnished_Status'].value_counts()
    st.write("Distribution of Furnishing Status:")
    st.write(furnishing_status_counts)
    fig_furnish_dist, ax_furnish_dist = plt.subplots(figsize=(8, 6))
    sns.barplot(x=furnishing_status_counts.index, y=furnishing_status_counts.values, ax=ax_furnish_dist)
    ax_furnish_dist.set_title('Distribution of Furnishing Status')
    ax_furnish_dist.set_xlabel('Furnishing Status')
    ax_furnish_dist.set_ylabel('Number of Properties')
    plt.tight_layout()
    st.pyplot(fig_furnish_dist)

    # Q13 (Public Transport Accessibility & Parking Space)
    st.subheader('Accessibility: Public Transport and Parking Space')
    public_transport_counts = df_eda['Public_Transport_Accessibility'].value_counts()
    st.write("Distribution of Public_Transport_Accessibility:")
    st.write(public_transport_counts)

    parking_space_counts = df_eda['Parking_Space'].value_counts()
    st.write("\nDistribution of Parking_Space:")
    st.write(parking_space_counts)

    fig_acc_park, axes_acc_park = plt.subplots(1, 2, figsize=(12, 6))
    sns.barplot(x=public_transport_counts.index, y=public_transport_counts.values, ax=axes_acc_park[0])
    axes_acc_park[0].set_title('Public Transport Accessibility')
    axes_acc_park[0].set_xlabel('Accessibility Level')
    axes_acc_park[0].set_ylabel('Number of Properties')

    sns.barplot(x=parking_space_counts.index, y=parking_space_counts.values, ax=axes_acc_park[1])
    axes_acc_park[1].set_title('Parking Space Availability')
    axes_acc_park[1].set_xlabel('Parking Space Availability')
    axes_acc_park[1].set_ylabel('Number of Properties')
    plt.tight_layout()
    st.pyplot(fig_acc_park)

    # Q14 (Security & Amenities)
    st.subheader('Security and Amenities Insights')
    security_counts = df_eda['Security'].value_counts()
    st.write("Distribution of Security Status:")
    st.write(security_counts)

    fig_sec_amen, axes_sec_amen = plt.subplots(1, 2, figsize=(14, 7))
    sns.barplot(x=security_counts.index, y=security_counts.values, ax=axes_sec_amen[0])
    axes_sec_amen[0].set_title('Distribution of Security Status')
    axes_sec_amen[0].set_xlabel('Security Status')
    axes_sec_amen[0].set_ylabel('Number of Properties')

    # Amenities
    all_amenities = []
    for amenities_str in df_eda['Amenities']:
        amenities_list = [a.strip() for a in amenities_str.split(',')]
        all_amenities.extend(amenities_list)
    amenities_series = pd.Series(all_amenities)
    amenities_counts = amenities_series.value_counts()

    st.write("\nTotal number of unique amenities:", len(amenities_counts))
    st.write("\nTop 10 Most Common Amenities:")
    st.write(amenities_counts.head(10))

    average_amenities_per_property = df_eda['Amenities'].apply(lambda x: len(x.split(','))).mean()
    st.write(f"\nAverage number of amenities per property: {average_amenities_per_property:.2f}")

    sns.barplot(x=amenities_counts.head(10).index, y=amenities_counts.head(10).values, ax=axes_sec_amen[1])
    axes_sec_amen[1].set_title('Top 10 Most Common Amenities')
    axes_sec_amen[1].set_xlabel('Amenity')
    axes_sec_amen[1].set_ylabel('Number of Properties')
    axes_sec_amen[1].set_xticklabels(axes_sec_amen[1].get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_sec_amen)

    # Q15 (Facing & Owner Type)
    st.subheader('Facing Direction and Owner Type')
    facing_counts = df_eda['Facing'].value_counts()
    st.write("Distribution of Facing directions:")
    st.write(facing_counts)

    owner_type_counts = df_eda['Owner_Type'].value_counts()
    st.write("\nDistribution of Owner Type:")
    st.write(owner_type_counts)

    fig_face_owner, axes_face_owner = plt.subplots(1, 2, figsize=(12, 6))
    sns.barplot(x=facing_counts.index, y=facing_counts.values, ax=axes_face_owner[0])
    axes_face_owner[0].set_title('Distribution of Facing Directions')
    axes_face_owner[0].set_xlabel('Facing Direction')
    axes_face_owner[0].set_ylabel('Number of Properties')

    sns.barplot(x=owner_type_counts.index, y=owner_type_counts.values, ax=axes_face_owner[1])
    axes_face_owner[1].set_title('Distribution of Owner Type')
    axes_face_owner[1].set_xlabel('Owner Type')
    axes_face_owner[1].set_ylabel('Number of Properties')
    plt.tight_layout()
    st.pyplot(fig_face_owner)

    # Q16 (Availability Status)
    st.subheader('Availability Status')
    availability_status_counts = df_eda['Availability_Status'].value_counts()
    st.write("Distribution of Availability Status:")
    st.write(availability_status_counts)
    fig_avail_dist, ax_avail_dist = plt.subplots(figsize=(8, 6))
    sns.barplot(x=availability_status_counts.index, y=availability_status_counts.values, ax=ax_avail_dist)
    ax_avail_dist.set_title('Distribution of Availability Status')
    ax_avail_dist.set_xlabel('Availability Status')
    ax_avail_dist.set_ylabel('Number of Properties')
    plt.tight_layout()
    st.pyplot(fig_avail_dist)

    st.header('Location-based Analysis')

    # Q6 (Average Price per SqFt by State)
    st.subheader('Average Price per SqFt by State')
    avg_price_per_sqft_by_state = df_eda.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False)
    st.write("Top 10 States by Average Price per SqFt:")
    st.write(avg_price_per_sqft_by_state.head(10))
    fig_avg_sqft_state, ax_avg_sqft_state = plt.subplots(figsize=(14, 7))
    sns.barplot(x=avg_price_per_sqft_by_state.head(10).index, y=avg_price_per_sqft_by_state.head(10).values, ax=ax_avg_sqft_state)
    ax_avg_sqft_state.set_title('Top 10 States by Average Price per SqFt')
    ax_avg_sqft_state.set_xlabel('State')
    ax_avg_sqft_state.set_ylabel('Average Price per SqFt')
    ax_avg_sqft_state.set_xticklabels(ax_avg_sqft_state.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_avg_sqft_state)

    # Q7 (Average Property Price by City)
    st.subheader('Average Property Price by City')
    avg_price_by_city = df_eda.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False)
    st.write("Top 10 Cities by Average Property Price:")
    st.write(avg_price_by_city.head(10))
    fig_avg_price_city, ax_avg_price_city = plt.subplots(figsize=(14, 7))
    sns.barplot(x=avg_price_by_city.head(10).index, y=avg_price_by_city.head(10).values, ax=ax_avg_price_city)
    ax_avg_price_city.set_title('Top 10 Cities by Average Property Price')
    ax_avg_price_city.set_xlabel('City')
    ax_avg_price_city.set_ylabel('Average Price in Lakhs')
    ax_avg_price_city.set_xticklabels(ax_avg_price_city.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_avg_price_city)

    # Q8 (Price Trend Across All Cities)
    st.subheader('Price Trend Across All Cities')
    city_price_trend = df_eda.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False)
    fig_city_price_trend, ax_city_price_trend = plt.subplots(figsize=(12, 7))
    sns.lineplot(x=city_price_trend.index, y=city_price_trend.values, marker='o', ax=ax_city_price_trend)
    ax_city_price_trend.set_title('Average Price_in_Lakhs by City')
    ax_city_price_trend.set_xlabel('City')
    ax_city_price_trend.set_ylabel('Average Price_in_Lakhs')
    ax_city_price_trend.set_xticklabels(ax_city_price_trend.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_city_price_trend)

    # Q9 (Price Trend Across All States)
    st.subheader('Price Trend Across All States')
    state_price_trend = df_eda.groupby('State')['Price_in_Lakhs'].mean().sort_values(ascending=False)
    fig_state_price_trend, ax_state_price_trend = plt.subplots(figsize=(9, 10))
    sns.lineplot(x=state_price_trend.index, y=state_price_trend.values, marker='o', ax=ax_state_price_trend)
    ax_state_price_trend.set_title('Average Price_in_Lakhs by State')
    ax_state_price_trend.set_xlabel('State')
    ax_state_price_trend.set_ylabel('Average Price_in_Lakhs')
    ax_state_price_trend.set_xticklabels(ax_state_price_trend.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_state_price_trend)

    # Q10 (Median Age by Locality)
    st.subheader('Median Age of Properties by Locality')
    median_age_by_locality = df_eda.groupby('Locality')['Age_of_Property'].median().sort_values(ascending=False)
    st.write("Top 10 Localities by Median Age of Property:")
    st.write(median_age_by_locality.head(10))
    fig_median_age_locality, ax_median_age_locality = plt.subplots(figsize=(14, 7))
    sns.barplot(x=median_age_by_locality.head(10).index, y=median_age_by_locality.head(10).values, ax=ax_median_age_locality)
    ax_median_age_locality.set_title('Top 10 Localities by Median Age of Property')
    ax_median_age_locality.set_xlabel('Locality')
    ax_median_age_locality.set_ylabel('Median Age of Property')
    ax_median_age_locality.set_xticklabels(ax_median_age_locality.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_median_age_locality)

    # Q11 (BHK Distribution Across Cities)
    st.subheader('BHK Distribution Across Top Cities')
    # Assuming city_counts is already calculated in the previous categorical features section
    city_counts = df_eda['City'].value_counts() # Recalculate if not in scope
    top_cities = city_counts.head(10).index
    df_top_cities = df_eda[df_eda['City'].isin(top_cities)]
    fig_bhk_top_cities, ax_bhk_top_cities = plt.subplots(figsize=(15, 8))
    sns.countplot(data=df_top_cities, x='City', hue='BHK', palette='viridis', ax=ax_bhk_top_cities)
    ax_bhk_top_cities.set_title('BHK Distribution Across Top 10 Cities')
    ax_bhk_top_cities.set_xlabel('City')
    ax_bhk_top_cities.set_ylabel('Number of Properties')
    ax_bhk_top_cities.set_xticklabels(ax_bhk_top_cities.get_xticklabels(), rotation=45, ha='right')
    ax_bhk_top_cities.legend(title='BHK')
    plt.tight_layout()
    st.pyplot(fig_bhk_top_cities)

    # Q12 (Price Trends for Top 5 Most Expensive Localities)
    st.subheader('Price Trends for Top 5 Most Expensive Localities')
    avg_price_by_locality = df_eda.groupby('Locality')['Price_in_Lakhs'].mean().sort_values(ascending=False)
    top_5_expensive_localities = avg_price_by_locality.head(5)
    st.write("Top 5 Most Expensive Localities by Average Price in Lakhs:")
    st.write(top_5_expensive_localities)
    fig_top_5_loc_price, ax_top_5_loc_price = plt.subplots(figsize=(12, 7))
    sns.barplot(x=top_5_expensive_localities.index, y=top_5_expensive_localities.values, ax=ax_top_5_loc_price)
    ax_top_5_loc_price.set_title('Top 5 Localities by Average Price in Lakhs')
    ax_top_5_loc_price.set_xlabel('Locality')
    ax_top_5_loc_price.set_ylabel('Average Price in Lakhs')
    ax_top_5_loc_price.set_xticklabels(ax_top_5_loc_price.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_top_5_loc_price)

    st.header('Feature Relationship & Correlation')

    # Q13 (Numerical Feature Correlation)
    st.subheader('Correlation Heatmap of Numerical Features')
    numerical_cols = df_ml.select_dtypes(include=np.number).columns
    correlation_matrix = df_ml[numerical_cols].corr()

    fig_corr_heatmap, ax_corr_heatmap = plt.subplots(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr_heatmap)
    ax_corr_heatmap.set_title('Correlation Heatmap of Numerical Features', fontsize=16)
    ax_corr_heatmap.set_xticklabels(ax_corr_heatmap.get_xticklabels(), rotation=45, ha='right')
    ax_corr_heatmap.set_yticklabels(ax_corr_heatmap.get_yticklabels(), rotation=0)
    plt.tight_layout()
    st.pyplot(fig_corr_heatmap)

    # Q14 (Nearby Schools vs. Price per SqFt)
    st.subheader('Nearby Schools vs. Price per SqFt')
    fig_schools_price, ax_schools_price = plt.subplots(figsize=(12, 7))
    sns.boxplot(x='Nearby_Schools', y='Price_per_SqFt', data=df_ml, ax=ax_schools_price)
    ax_schools_price.set_title('Price per SqFt vs. Number of Nearby Schools')
    ax_schools_price.set_xlabel('Number of Nearby Schools')
    ax_schools_price.set_ylabel('Price per SqFt')
    ax_schools_price.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_schools_price)

    correlation_coefficient_schools_price = df_ml['Nearby_Schools'].corr(df_ml['Price_per_SqFt'])
    st.write(f"Pearson correlation coefficient between Nearby_Schools and Price_per_SqFt: {correlation_coefficient_schools_price:.2f}")

    # Q15 (Nearby Hospitals vs. Price per SqFt)
    st.subheader('Nearby Hospitals vs. Price per SqFt')
    fig_hospitals_price, ax_hospitals_price = plt.subplots(figsize=(12, 7))
    sns.boxplot(x='Nearby_Hospitals', y='Price_per_SqFt', data=df_ml, ax=ax_hospitals_price)
    ax_hospitals_price.set_title('Price per SqFt vs. Number of Nearby Hospitals')
    ax_hospitals_price.set_xlabel('Number of Nearby Hospitals')
    ax_hospitals_price.set_ylabel('Price per SqFt')
    ax_hospitals_price.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_hospitals_price)

    correlation_coefficient_hospitals_price = df_ml['Nearby_Hospitals'].corr(df_ml['Price_per_SqFt'])
    st.write(f"Pearson correlation coefficient between Nearby_Hospitals and Price_per_SqFt: {correlation_coefficient_hospitals_price:.2f}")

    # Q16 (Price by Furnished Status)
    st.subheader('Price Distribution by Furnished Status')
    fig_furnish_price, ax_furnish_price = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Furnished_Status', y='Price_in_Lakhs', data=df_eda, ax=ax_furnish_price)
    ax_furnish_price.set_title('Price in Lakhs Distribution by Furnished Status')
    ax_furnish_price.set_xlabel('Furnished Status')
    ax_furnish_price.set_ylabel('Price in Lakhs')
    ax_furnish_price.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_furnish_price)

    # Q17 (Price per SqFt by Facing Direction)
    st.subheader('Price per SqFt Distribution by Facing Direction')
    fig_facing_price, ax_facing_price = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Facing', y='Price_per_SqFt', data=df_eda, ax=ax_facing_price)
    ax_facing_price.set_title('Price per SqFt Distribution by Facing Direction')
    ax_facing_price.set_xlabel('Facing Direction')
    ax_facing_price.set_ylabel('Price per SqFt')
    ax_facing_price.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_facing_price)

    st.header('Investment / Amenities / Ownership Analysis')

    # Q18 (Properties per Owner Type)
    st.subheader('Properties per Owner Type')
    owner_type_counts = df_eda['Owner_Type'].value_counts()
    st.write("Count of properties for each Owner_Type:")
    st.write(owner_type_counts)
    fig_owner_type, ax_owner_type = plt.subplots(figsize=(10, 6))
    sns.barplot(x=owner_type_counts.index, y=owner_type_counts.values, ax=ax_owner_type)
    ax_owner_type.set_title('Distribution of Properties by Owner Type')
    ax_owner_type.set_xlabel('Owner Type')
    ax_owner_type.set_ylabel('Number of Properties')
    plt.tight_layout()
    st.pyplot(fig_owner_type)

    # Q19 (Properties per Availability Status)
    st.subheader('Properties per Availability Status')
    availability_status_counts = df_eda['Availability_Status'].value_counts()
    st.write("Count of properties for each Availability_Status:")
    st.write(availability_status_counts)
    fig_avail_status, ax_avail_status = plt.subplots(figsize=(10, 6))
    sns.barplot(x=availability_status_counts.index, y=availability_status_counts.values, ax=ax_avail_status)
    ax_avail_status.set_title('Distribution of Properties by Availability Status')
    ax_avail_status.set_xlabel('Availability Status')
    ax_avail_status.set_ylabel('Number of Properties')
    plt.tight_layout()
    st.pyplot(fig_avail_status)

def prediction_page():
    st.title('Prediction Model - Coming Soon!')
    st.write('This page will contain the full prediction model, including user inputs and model outputs.')
