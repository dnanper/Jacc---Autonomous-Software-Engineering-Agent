"""
Query analysis abstraction for the memory system.

Provides an interface for analyzing natural language queries to extract
structured information like temporal constraints.

This module supports:
- Temporal expression extraction ("last week", "yesterday", "in 2023")
- Date range calculation from natural language
- Multiple language support via dateparser
"""

import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TemporalConstraint(BaseModel):
    """
    Temporal constraint extracted from a query.

    Represents a time range with start and end dates.
    Both dates are inclusive.
    """
    
    start_date: datetime = Field(description="Start of the time range (inclusive)")
    end_date: datetime = Field(description="End of the time range (inclusive)")

    def __str__(self) -> str:
        return f"[{self.start_date.date()} to {self.end_date.date()}]"


class QueryAnalysis(BaseModel):
    """
    Result of analyzing a natural language query.

    Contains extracted structured information like temporal constraints.
    This can be extended to include entity mentions, intent classification, etc.
    """
    
    temporal_constraint: TemporalConstraint | None = Field(
        default=None, description="Extracted temporal constraint, if any"
    )
    
    # Future extensions:
    # entities: list[str] | None = Field(default=None, description="Entities mentioned in query")
    # intent: str | None = Field(default=None, description="Detected query intent")


class QueryAnalyzer(ABC):
    """
    Abstract base class for query analysis.

    Implementations analyze natural language queries to extract structured
    information like temporal constraints, entities, etc.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load the query analyzer model.

        This should be called during initialization to load the model
        and avoid cold start latency on first analyze() call.
        """
        pass

    @abstractmethod
    def analyze(self, query: str, reference_date: datetime | None = None) -> QueryAnalysis:
        """
        Analyze a natural language query.

        Args:
            query: Natural language query to analyze
            reference_date: Reference date for relative terms (defaults to now)

        Returns:
            QueryAnalysis containing extracted information
        """
        pass


class DateparserQueryAnalyzer(QueryAnalyzer):
    """
    Query analyzer using dateparser library.

    Uses dateparser to extract temporal expressions from natural language
    queries. Supports 200+ languages including English, Spanish, Italian,
    German, Dutch, Portuguese, and many others.

    This is the recommended analyzer for production use due to:
    - Broad language support
    - No ML model loading (fast startup)
    - Correct handling of relative dates
    """

    def __init__(self):
        """Initialize dateparser query analyzer."""
        self._dateparser = None

    def load(self) -> None:
        """Load dateparser (lazy import)."""
        if self._dateparser is not None:
            return
        
        try:
            import dateparser
            self._dateparser = dateparser
            logger.info("QueryAnalyzer: dateparser loaded")
        except ImportError:
            raise ImportError(
                "dateparser is required for DateparserQueryAnalyzer. "
                "Install it with: pip install dateparser"
            )

    def analyze(self, query: str, reference_date: datetime | None = None) -> QueryAnalysis:
        """
        Analyze query using dateparser.

        Extracts temporal expressions from the query text. Supports multiple
        languages automatically.

        Args:
            query: Natural language query (any language)
            reference_date: Reference date for relative terms (defaults to now)

        Returns:
            QueryAnalysis with temporal_constraint if found
        """
        if self._dateparser is None:
            self.load()

        reference_date = reference_date or datetime.now()
        
        # First check for period patterns (last week, this month, etc.)
        period_constraint = self._extract_period(query, reference_date)
        if period_constraint:
            return QueryAnalysis(temporal_constraint=period_constraint)

        # Try to parse single date from query
        # dateparser settings for accurate relative date parsing
        settings = {
            "PREFER_DATES_FROM": "past",
            "RELATIVE_BASE": reference_date,
            "RETURN_AS_TIMEZONE_AWARE": False,
        }

        parsed_date = self._dateparser.parse(query, settings=settings)
        
        if parsed_date:
            # For single dates, create a day-long range
            start = datetime(parsed_date.year, parsed_date.month, parsed_date.day, 0, 0, 0)
            end = datetime(parsed_date.year, parsed_date.month, parsed_date.day, 23, 59, 59)
            return QueryAnalysis(
                temporal_constraint=TemporalConstraint(start_date=start, end_date=end)
            )

        return QueryAnalysis(temporal_constraint=None)

    def _extract_period(self, query: str, reference_date: datetime) -> TemporalConstraint | None:
        """
        Extract period-based temporal expressions (week, month, year, weekend).

        These need special handling as they represent date ranges, not single dates.
        Supports multiple languages.
        """
        query_lower = query.lower()

        def constraint(start: datetime, end: datetime) -> TemporalConstraint:
            return TemporalConstraint(
                start_date=start.replace(hour=0, minute=0, second=0, microsecond=0),
                end_date=end.replace(hour=23, minute=59, second=59, microsecond=999999),
            )

        # Weekday index (0=Monday, 6=Sunday)
        weekday = reference_date.weekday()

        # === Last week patterns (multiple languages) ===
        last_week_patterns = [
            "last week", "previous week",  # English
            "la semana pasada", "semana pasada",  # Spanish
            "la settimana scorsa", "settimana scorsa",  # Italian
            "letzte woche", "vergangene woche",  # German
            "vorige week", "afgelopen week",  # Dutch
            "semaine dernière", "la semaine dernière",  # French
            "semana passada", "a semana passada",  # Portuguese
            "tuần trước",  # Vietnamese
        ]
        if any(pattern in query_lower for pattern in last_week_patterns):
            # Monday to Sunday of last week
            days_since_monday = weekday
            last_monday = reference_date - timedelta(days=days_since_monday + 7)
            last_sunday = last_monday + timedelta(days=6)
            return constraint(last_monday, last_sunday)

        # === This week patterns ===
        this_week_patterns = [
            "this week", "current week",  # English
            "esta semana",  # Spanish
            "questa settimana",  # Italian
            "diese woche",  # German
            "cette semaine",  # French
            "esta semana",  # Portuguese
            "tuần này",  # Vietnamese
        ]
        if any(pattern in query_lower for pattern in this_week_patterns):
            # Monday to Sunday of this week
            days_since_monday = weekday
            this_monday = reference_date - timedelta(days=days_since_monday)
            this_sunday = this_monday + timedelta(days=6)
            return constraint(this_monday, this_sunday)

        # === Last weekend patterns ===
        last_weekend_patterns = [
            "last weekend", "previous weekend",  # English
            "el fin de semana pasado",  # Spanish
            "lo scorso fine settimana",  # Italian
            "letztes wochenende",  # German
            "dernier week-end", "le week-end dernier",  # French
            "fim de semana passado",  # Portuguese
            "cuối tuần trước",  # Vietnamese
        ]
        if any(pattern in query_lower for pattern in last_weekend_patterns):
            # Saturday and Sunday of last week
            days_until_last_saturday = weekday + 2 if weekday < 5 else weekday - 5
            last_saturday = reference_date - timedelta(days=days_until_last_saturday + 7)
            last_sunday = last_saturday + timedelta(days=1)
            return constraint(last_saturday, last_sunday)

        # === This weekend patterns ===
        this_weekend_patterns = [
            "this weekend",  # English
            "este fin de semana",  # Spanish
            "questo fine settimana",  # Italian
            "dieses wochenende",  # German
            "ce week-end",  # French
            "este fim de semana",  # Portuguese
            "cuối tuần này",  # Vietnamese
        ]
        if any(pattern in query_lower for pattern in this_weekend_patterns):
            # Saturday and Sunday of this week
            days_until_saturday = (5 - weekday) % 7
            this_saturday = reference_date + timedelta(days=days_until_saturday)
            this_sunday = this_saturday + timedelta(days=1)
            return constraint(this_saturday, this_sunday)

        # === Last month patterns ===
        last_month_patterns = [
            "last month", "previous month",  # English
            "el mes pasado",  # Spanish
            "il mese scorso",  # Italian
            "letzten monat", "vergangenen monat",  # German
            "le mois dernier",  # French
            "mês passado", "o mês passado",  # Portuguese
            "tháng trước",  # Vietnamese
        ]
        if any(pattern in query_lower for pattern in last_month_patterns):
            first_of_this_month = reference_date.replace(day=1)
            last_day_of_prev_month = first_of_this_month - timedelta(days=1)
            first_of_prev_month = last_day_of_prev_month.replace(day=1)
            return constraint(first_of_prev_month, last_day_of_prev_month)

        # === This month patterns ===
        this_month_patterns = [
            "this month", "current month",  # English
            "este mes",  # Spanish
            "questo mese",  # Italian
            "diesen monat",  # German
            "ce mois", "ce mois-ci",  # French
            "este mês",  # Portuguese
            "tháng này",  # Vietnamese
        ]
        if any(pattern in query_lower for pattern in this_month_patterns):
            first_of_this_month = reference_date.replace(day=1)
            # Find last day of current month
            if reference_date.month == 12:
                next_month = reference_date.replace(year=reference_date.year + 1, month=1, day=1)
            else:
                next_month = reference_date.replace(month=reference_date.month + 1, day=1)
            last_of_this_month = next_month - timedelta(days=1)
            return constraint(first_of_this_month, last_of_this_month)

        # === Last year patterns ===
        last_year_patterns = [
            "last year", "previous year",  # English
            "el año pasado",  # Spanish
            "l'anno scorso",  # Italian
            "letztes jahr", "vergangenes jahr",  # German
            "l'année dernière", "l'an dernier",  # French
            "ano passado", "o ano passado",  # Portuguese
            "năm ngoái",  # Vietnamese
        ]
        if any(pattern in query_lower for pattern in last_year_patterns):
            last_year = reference_date.year - 1
            return constraint(
                datetime(last_year, 1, 1),
                datetime(last_year, 12, 31),
            )

        # === This year patterns ===
        this_year_patterns = [
            "this year", "current year",  # English
            "este año",  # Spanish
            "quest'anno",  # Italian
            "dieses jahr",  # German
            "cette année",  # French
            "este ano",  # Portuguese
            "năm nay",  # Vietnamese
        ]
        if any(pattern in query_lower for pattern in this_year_patterns):
            return constraint(
                datetime(reference_date.year, 1, 1),
                datetime(reference_date.year, 12, 31),
            )

        # === Yesterday patterns ===
        yesterday_patterns = ["yesterday", "ayer", "ieri", "gestern", "hier", "ontem", "hôm qua"]
        if any(pattern in query_lower for pattern in yesterday_patterns):
            yesterday = reference_date - timedelta(days=1)
            return constraint(yesterday, yesterday)

        # === Today patterns ===
        today_patterns = ["today", "hoy", "oggi", "heute", "aujourd'hui", "hoje", "hôm nay"]
        if any(pattern in query_lower for pattern in today_patterns):
            return constraint(reference_date, reference_date)

        # === Specific year patterns (e.g., "in 2023", "in 2024") ===
        year_match = re.search(r"\b(in\s+)?(\d{4})\b", query_lower)
        if year_match:
            year = int(year_match.group(2))
            if 1900 <= year <= 2100:  # Reasonable year range
                return constraint(
                    datetime(year, 1, 1),
                    datetime(year, 12, 31),
                )

        return None


# Factory function
def create_query_analyzer() -> QueryAnalyzer:
    """
    Create the default query analyzer.
    
    Returns:
        DateparserQueryAnalyzer instance (recommended for production)
    """
    return DateparserQueryAnalyzer()
