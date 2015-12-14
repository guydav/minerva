from peewee import *

database = MySQLDatabase('pitchfx', **{})


class UnknownField(object):
    pass


class BaseModel(Model):
    class Meta:
        database = database


class Umpiretmp2(BaseModel):
    ball = CharField()
    date = DateField()
    des = CharField()
    px = FloatField(null=True)
    pz = FloatField(null=True)
    stand = CharField()
    strike = CharField()
    sz_bot = FloatField(null=True)
    sz_top = FloatField(null=True)
    throws = CharField(null=True)
    type = CharField()

    class Meta:
        db_table = 'UMPIRETMP2'


class Player(BaseModel):
    eliasid = PrimaryKeyField()
    first = CharField()
    height = IntegerField(null=True)
    lahmanid = CharField(null=True)
    last = CharField()
    throws = CharField(index=True, null=True)

    class Meta:
        db_table = 'players'
        indexes = (
            (('first', 'last'), False),
        )


class PlayerCopy(BaseModel):
    eliasid = PrimaryKeyField()
    first = CharField()
    height = IntegerField(null=True)
    lahmanid = CharField(null=True)
    last = CharField()
    throws = CharField(index=True, null=True)

    class Meta:
        db_table = 'players_copy'
        indexes = (
            (('first', 'last'), False),
        )


class Team(BaseModel):
    city = CharField(null=True)
    name = CharField(null=True)
    team = CharField(primary_key=True)

    class Meta:
        db_table = 'teams'


class Umpire(BaseModel):
    first = CharField()
    last = CharField()
    ump = PrimaryKeyField(db_column='ump_id')

    class Meta:
        db_table = 'umpires'
        indexes = (
            (('first', 'last'), True),
        )


class GameType(BaseModel):
    type = CharField()

    class Meta:
        db_table = 'game_types'


class Game(BaseModel):
    # away = CharField()
    away = ForeignKeyField(Team, db_column='away', related_name='away_games')
    away_starting_pitcher = IntegerField(null=True)
    date = DateField(index=True)
    game = IntegerField()
    game_id = PrimaryKeyField()
    # home = CharField()
    home = ForeignKeyField(Team, db_column='home', related_name='home_games')
    home_starting_pitcher = IntegerField(null=True)
    local_time = TimeField(null=True)
    runs_away = IntegerField(null=True)
    runs_home = IntegerField(null=True)
    temp = IntegerField(null=True)
    # type = IntegerField()
    type = ForeignKeyField(GameType, db_column='type', related_name='games')
    # umpire = IntegerField(null=True)
    umpire = ForeignKeyField(Umpire, db_column='umpire', related_name='games')
    wind = IntegerField(null=True)
    wind_dir = CharField(null=True)

    class Meta:
        db_table = 'games'


class AtBat(BaseModel):
    ab = PrimaryKeyField(db_column='ab_id')
    ball = CharField()
    # batter = IntegerField()
    batter = ForeignKeyField(Player, db_column='batter', related_name='at_bats_batted')
    des = CharField()
    event = CharField()
    # game_id = IntegerField(db_column='game_id', index=True)
    game = ForeignKeyField(Game, related_name='at_bats', index=True)
    hit_type = CharField(null=True)
    hit_x = FloatField(null=True)
    hit_y = FloatField(null=True)
    inning = CharField()
    num = IntegerField()
    outs = CharField()
    # pitcher = IntegerField()
    pitcher = ForeignKeyField(Player, db_column='pitcher', related_name='at_bats_pitched')
    stand = CharField()
    strike = CharField()

    class Meta:
        db_table = 'atbats'


class PitchType(BaseModel):
    id = CharField(primary_key=True)
    pitch = CharField()

    class Meta:
        db_table = 'pitch_types'


class Pitch(BaseModel):
    # ab_id = IntegerField(db_column='ab_id', index=True)
    ab = ForeignKeyField(AtBat, related_name='pitches', index=True)
    ax = FloatField(null=True)
    ay = FloatField(null=True)
    az = FloatField(null=True)
    ball = CharField(null=True)
    break_angle = FloatField(null=True)
    break_length = FloatField(null=True)
    break_y = FloatField(null=True)
    des = CharField()
    end_speed = FloatField(null=True)
    id = IntegerField()
    my_pitch_type = IntegerField(null=True)
    on_1b = IntegerField(null=True)
    on_2b = IntegerField(null=True)
    on_3b = IntegerField(null=True)
    pfx_x = FloatField(null=True)
    pfx_z = FloatField(null=True)
    pitch = PrimaryKeyField(db_column='pitch_id')
    # pitch_type = CharField(null=True)
    pitch_type = ForeignKeyField(PitchType, related_name='pitches')
    px = FloatField(null=True)
    pz = FloatField(null=True)
    start_speed = FloatField(null=True)
    strike = CharField(null=True)
    sv = CharField(db_column='sv_id', null=True)
    sz_bot = FloatField(null=True)
    sz_top = FloatField(null=True)
    type = CharField()
    type_confidence = FloatField(null=True)
    vx0 = FloatField(null=True)
    vy0 = FloatField(null=True)
    vz0 = FloatField(null=True)
    x = FloatField()
    x0 = FloatField(null=True)
    y = FloatField()
    y0 = FloatField(null=True)
    z0 = FloatField(null=True)

    class Meta:
        db_table = 'pitches'


class Average(BaseModel):
    avg = FloatField(null=True)
    phand = CharField()
    scale = IntegerField()
    type = CharField()
    x1 = DecimalField()
    x2 = DecimalField()
    y1 = DecimalField()
    y2 = DecimalField()

    class Meta:
        db_table = 'averages'
        indexes = (
            (('type', 'phand', 'scale', 'x1', 'x2', 'y1', 'y2'), True),
        )
        primary_key = CompositeKey('phand', 'scale', 'type', 'x1', 'x2', 'y1', 'y2')


class Delete4Bill(BaseModel):
    pitchid = IntegerField(null=True)

    class Meta:
        db_table = 'delete_4Bill'


class Height(BaseModel):
    eliasid = PrimaryKeyField()
    height2 = IntegerField(null=True)

    class Meta:
        db_table = 'height'


class PitchTransition(BaseModel):
    # ab = IntegerField(db_column='ab_id', index=True)
    ab = ForeignKeyField(AtBat, related_name='pitch_transitions')
    first_ball = CharField(null=True)
    first_des = CharField()
    first = IntegerField(db_column='first_id')
    first_on_1b = IntegerField(null=True)
    first_on_2b = IntegerField(null=True)
    first_on_3b = IntegerField(null=True)
    # first_pitch = IntegerField(db_column='first_pitch_id', index=True)
    first_pitch = ForeignKeyField(Pitch, related_name='transitions_as_first_pitch', index=True)
    # first_pitch_type = CharField(null=True)
    first_pitch_type = ForeignKeyField(PitchType, related_name='transitions_as_first_pitch')
    first_strike = CharField(null=True)
    first_sv = CharField(db_column='first_sv_id', null=True)
    first_type = CharField()
    first_type_confidence = FloatField(null=True)
    second_ball = CharField(null=True)
    second_des = CharField()
    second = IntegerField(db_column='second_id')
    second_on_1b = IntegerField(null=True)
    second_on_2b = IntegerField(null=True)
    second_on_3b = IntegerField(null=True)
    # second_pitch = IntegerField(db_column='second_pitch_id', index=True)
    second_pitch = ForeignKeyField(Pitch, related_name='transitions_as_second_pitch', index=True)
    # second_pitch_type = CharField(null=True)
    second_pitch_type = ForeignKeyField(PitchType, related_name='transitions_as_second_pitch')
    second_strike = CharField(null=True)
    second_sv = CharField(db_column='second_sv_id', null=True)
    second_type = CharField()
    second_type_confidence = FloatField(null=True)
    transition = PrimaryKeyField(db_column='transition_id')

    class Meta:
        db_table = 'pitch_transitions'
